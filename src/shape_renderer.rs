use std::ops::Range;

use wgpu::{
    BlendState, Buffer, BufferBinding, BufferUsages, ColorTargetState, Device, MultisampleState, PipelineCache,
    PrimitiveState, RenderPass, RenderPipeline, RenderPipelineDescriptor, TextureFormat, VertexStepMode,
    util::{BufferInitDescriptor, DeviceExt as _},
};
use winit::dpi::PhysicalSize;

use crate::shape_shaders::shape::{self, InstanceInput, VertexInput};

pub struct ShapeRenderer {
    vertex_buffer: Buffer,
    instance_buffer: Option<Buffer>,
    instance_count: u32,
    uniforms_bind_group: Option<shape::WgpuBindGroup0>,
    render_pipeline: RenderPipeline,
}

impl ShapeRenderer {
    pub fn new(device: &Device, swapchain_format: TextureFormat, pipeline_cache: &PipelineCache) -> Self {
        let pipeline_layout = shape::create_pipeline_layout(device);
        let shader = shape::create_shader_module_embed_source(device);

        let vertex_entry = shape::vs_main_entry(VertexStepMode::Vertex, VertexStepMode::Instance);
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

        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Shape vertex buffer"),
            contents: bytemuck::cast_slice(&[
                VertexInput::new([1.0, 1.0]),
                VertexInput::new([-1.0, 1.0]),
                VertexInput::new([-1.0, -1.0]),
                VertexInput::new([-1.0, -1.0]),
                VertexInput::new([1.0, -1.0]),
                VertexInput::new([1.0, 1.0]),
            ]),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        Self {
            vertex_buffer,
            instance_buffer: None,
            instance_count: 0,
            render_pipeline,
            uniforms_bind_group: None,
        }
    }

    pub fn prepare(&mut self, device: &Device, instances: &[InstanceInput], viewport_size: PhysicalSize<u32>) {
        let uniforms = shape::Uniforms::new(viewport_size.into());
        let uniforms_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Transform buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        self.uniforms_bind_group = Some(shape::WgpuBindGroup0::from_bindings(
            device,
            shape::WgpuBindGroup0Entries::new(shape::WgpuBindGroup0EntriesParams {
                uniforms: BufferBinding {
                    buffer: &uniforms_buffer,
                    offset: 0,
                    size: None,
                },
            }),
        ));

        self.instance_buffer = self
            .instance_buffer
            .take()
            .filter(|buffer| {
                usize::try_from(buffer.size()).is_ok_and(|buffer_size| buffer_size >= size_of_val(instances))
            })
            .or_else(|| {
                Some(device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Shape instance buffer"),
                    contents: bytemuck::cast_slice(instances),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                }))
            });
        self.instance_count = u32::try_from(instances.len()).expect("Too many instances");
    }

    pub fn render(&self, render_pass: &mut RenderPass<'_>, instances: Range<usize>) {
        let Some((instance_buffer, uniforms_bind_group)) =
            self.instance_buffer.as_ref().zip(self.uniforms_bind_group.as_ref())
        else {
            panic!("Forgot to call prepare()?")
        };
        render_pass.set_pipeline(&self.render_pipeline);
        uniforms_bind_group.set(render_pass);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
        let start = u32::try_from(instances.start).unwrap();
        let end = u32::try_from(instances.end).unwrap();
        render_pass.draw(0..6, start..end);
    }
}
