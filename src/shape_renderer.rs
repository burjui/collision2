use std::ops::Range;

use wgpu::{
    BlendState, BufferBinding, BufferUsages, ColorTargetState, Device, MultisampleState, PipelineCache, PrimitiveState,
    Queue, RenderPass, RenderPipeline, RenderPipelineDescriptor, TextureFormat, VertexStepMode,
};
use winit::dpi::PhysicalSize;

use crate::{
    objects::Objects,
    shape_shaders::shape::{self, ColorInput, FlagsInput, PositionInput, ShapeInput, SizeInput, VertexInput},
    wgpu_buffer::WgpuBuffer,
};

struct Buffers {
    flag: WgpuBuffer<shape::FlagsInput>,
    position: WgpuBuffer<shape::PositionInput>,
    size: WgpuBuffer<shape::SizeInput>,
    color: WgpuBuffer<shape::ColorInput>,
    shape: WgpuBuffer<shape::ShapeInput>,
}

impl Buffers {
    fn len(&self) -> usize {
        self.flag.len()
    }
}

pub struct ShapeRenderer {
    vertex_buffer: WgpuBuffer<VertexInput>,
    render_pipeline: RenderPipeline,
    uniforms_buffer: WgpuBuffer<shape::Uniforms>,
    uniforms_bind_group: shape::WgpuBindGroup0,
    instance_buffers: Option<Buffers>,
    instance_count: usize,
}

impl ShapeRenderer {
    pub fn new(
        device: &Device,
        queue: &Queue,
        swapchain_format: TextureFormat,
        pipeline_cache: &PipelineCache,
    ) -> Self {
        let vertex_buffer =
            WgpuBuffer::new(device, "Shape vertex buffer", 6, BufferUsages::VERTEX | BufferUsages::COPY_DST);
        vertex_buffer.enque_write(
            queue,
            bytemuck::cast_slice(&[
                VertexInput::new([1.0, 1.0]),
                VertexInput::new([-1.0, 1.0]),
                VertexInput::new([-1.0, -1.0]),
                VertexInput::new([-1.0, -1.0]),
                VertexInput::new([1.0, -1.0]),
                VertexInput::new([1.0, 1.0]),
            ]),
        );

        let pipeline_layout = shape::create_pipeline_layout(device);
        let shader = shape::create_shader_module_embed_source(device);

        let vertex_entry = shape::vs_main_entry(
            VertexStepMode::Vertex,
            VertexStepMode::Instance,
            VertexStepMode::Instance,
            VertexStepMode::Instance,
            VertexStepMode::Instance,
            VertexStepMode::Instance,
        );
        let vertex_state = shape::vertex_state(&shader, &vertex_entry);

        let color_target_state = ColorTargetState {
            blend: Some(BlendState::ALPHA_BLENDING),
            ..ColorTargetState::from(swapchain_format)
        };
        let fragment_entry = shape::fs_main_entry([Some(color_target_state)]);
        let fragment_state = shape::fragment_state(&shader, &fragment_entry);

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: vertex_state,
            fragment: Some(fragment_state),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: Some(pipeline_cache),
        });

        let uniforms_buffer =
            WgpuBuffer::new(device, "Uniform buffer", 1, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let uniforms_bind_group = shape::WgpuBindGroup0::from_bindings(
            device,
            shape::WgpuBindGroup0Entries::new(shape::WgpuBindGroup0EntriesParams {
                uniforms: BufferBinding {
                    buffer: uniforms_buffer.inner(),
                    offset: 0,
                    size: None,
                },
            }),
        );

        Self {
            vertex_buffer,
            render_pipeline,
            uniforms_buffer,
            uniforms_bind_group,
            instance_buffers: None,
            instance_count: 0,
        }
    }

    pub fn prepare(&mut self, device: &Device, queue: &Queue, objects: &Objects, viewport_size: PhysicalSize<u32>) {
        self.uniforms_buffer.enque_write(queue, bytemuck::cast_slice(&[shape::Uniforms::new(viewport_size.into())]));

        let instance_buffers =
            self.instance_buffers.take().filter(|buffers| buffers.len() >= objects.len()).unwrap_or_else(|| {
                let usage = BufferUsages::VERTEX | BufferUsages::COPY_DST;
                Buffers {
                    flag: WgpuBuffer::new(device, "Shape flag instance buffer", objects.len(), usage),
                    position: WgpuBuffer::new(device, "Shape position instance buffer", objects.len(), usage),
                    size: WgpuBuffer::new(device, "Shape size instance buffer", objects.len(), usage),
                    color: WgpuBuffer::new(device, "Shape color instance buffer", objects.len(), usage),
                    shape: WgpuBuffer::new(device, "Shape instance buffer", objects.len(), usage),
                }
            });
        instance_buffers.flag.enque_write(queue, &objects.flags);
        instance_buffers.position.enque_write(queue, &objects.position);
        instance_buffers.size.enque_write(queue, &objects.size);
        instance_buffers.color.enque_write(queue, &objects.color);
        instance_buffers.shape.enque_write(queue, &objects.shape);
        self.instance_buffers = Some(instance_buffers);
        self.instance_count = objects.len();
    }

    pub fn render(&self, render_pass: &mut RenderPass<'_>, instances: Range<usize>) {
        let Some(instance_buffers) = self.instance_buffers.as_ref() else {
            panic!("Forgot to call prepare()?")
        };
        render_pass.set_pipeline(&self.render_pipeline);
        self.uniforms_bind_group.set(render_pass);
        render_pass
            .set_vertex_buffer(VertexInput::VERTEX_ATTRIBUTES[0].shader_location, self.vertex_buffer.inner().slice(..));
        render_pass.set_vertex_buffer(
            FlagsInput::VERTEX_ATTRIBUTES[0].shader_location,
            instance_buffers.flag.inner().slice(..),
        );
        render_pass.set_vertex_buffer(
            PositionInput::VERTEX_ATTRIBUTES[0].shader_location,
            instance_buffers.position.inner().slice(..),
        );
        render_pass.set_vertex_buffer(
            SizeInput::VERTEX_ATTRIBUTES[0].shader_location,
            instance_buffers.size.inner().slice(..),
        );
        render_pass.set_vertex_buffer(
            ColorInput::VERTEX_ATTRIBUTES[0].shader_location,
            instance_buffers.color.inner().slice(..),
        );
        render_pass.set_vertex_buffer(
            ShapeInput::VERTEX_ATTRIBUTES[0].shader_location,
            instance_buffers.shape.inner().slice(..),
        );
        let start = u32::try_from(instances.start).unwrap();
        let end = u32::try_from(instances.end).unwrap();
        render_pass.draw(0..6, start..end);
    }
}
