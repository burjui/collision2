use std::ops::Range;

use wgpu::{
    BlendState, BufferBinding, BufferUsages, ColorTargetState, Device, MultisampleState, PipelineCache, PrimitiveState,
    Queue, RenderPass, RenderPipeline, RenderPipelineDescriptor, TextureFormat, VertexStepMode,
};
use winit::dpi::PhysicalSize;

use crate::{
    gpu_arena::{GpuArena, GpuSlice},
    objects::Objects,
    shape_shaders::shape::{self, ColorInput, FlagsInput, PositionInput, ShapeInput, SizeInput, VertexInput},
};

struct InstanceBuffers {
    flags: GpuSlice<shape::FlagsInput>,
    position: GpuSlice<shape::PositionInput>,
    size: GpuSlice<shape::SizeInput>,
    color: GpuSlice<shape::ColorInput>,
    shape: GpuSlice<shape::ShapeInput>,
}

impl InstanceBuffers {
    fn len(&self) -> usize {
        self.flags.len()
    }
}

pub struct ShapeRenderer {
    vertex_buffer: GpuSlice<VertexInput>,
    render_pipeline: RenderPipeline,
    uniforms_buffer: GpuSlice<shape::Uniforms>,
    uniforms_bind_group: shape::WgpuBindGroup0,
    instance_buffers: Option<InstanceBuffers>,
    instance_count: usize,
}

impl ShapeRenderer {
    pub fn new(
        device: &Device,
        queue: &Queue,
        swapchain_format: TextureFormat,
        pipeline_cache: &PipelineCache,
    ) -> Self {
        let (_, vertex_buffer) =
            GpuArena::new_slice(6, "Shape vertex arena", BufferUsages::VERTEX | BufferUsages::COPY_DST, device);
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

        let (uniforms_arena, uniforms_buffer) =
            GpuArena::new_slice(1, "Shape uniforms arena", BufferUsages::UNIFORM | BufferUsages::COPY_DST, device);
        let uniforms_bind_group = shape::WgpuBindGroup0::from_bindings(
            device,
            shape::WgpuBindGroup0Entries::new(shape::WgpuBindGroup0EntriesParams {
                uniforms: BufferBinding {
                    buffer: uniforms_arena.buffer(),
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
                let mut instance_arena = GpuArena::new(
                    objects.len()
                        * (size_of::<shape::FlagsInput>()
                            + size_of::<shape::PositionInput>()
                            + size_of::<shape::SizeInput>()
                            + size_of::<shape::ColorInput>()
                            + size_of::<shape::ShapeInput>()),
                    "Shape instance arena",
                    BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    device,
                );
                println!("instance arena size: {}", instance_arena.buffer().size());
                InstanceBuffers {
                    flags: instance_arena.allocate(objects.len()),
                    position: instance_arena.allocate(objects.len()),
                    size: instance_arena.allocate(objects.len()),
                    color: instance_arena.allocate(objects.len()),
                    shape: instance_arena.allocate(objects.len()),
                }
            });
        instance_buffers.flags.enque_write(queue, &objects.flags);
        instance_buffers.position.enque_write(queue, &objects.position);
        instance_buffers.size.enque_write(queue, &objects.size);
        instance_buffers.color.enque_write(queue, &objects.color);
        instance_buffers.shape.enque_write(queue, &objects.shape);
        self.instance_buffers = Some(instance_buffers);
        self.instance_count = objects.len();
    }

    pub fn render(&self, rpass: &mut RenderPass<'_>, instances: Range<usize>) {
        let Some(InstanceBuffers {
            flags,
            position,
            size,
            color,
            shape,
        }) = self.instance_buffers.as_ref()
        else {
            panic!("Forgot to call prepare()?")
        };
        rpass.set_pipeline(&self.render_pipeline);
        self.uniforms_bind_group.set(rpass);

        rpass.set_vertex_buffer(VertexInput::VERTEX_ATTRIBUTES[0].shader_location, self.vertex_buffer.as_slice());
        rpass.set_vertex_buffer(FlagsInput::VERTEX_ATTRIBUTES[0].shader_location, flags.slice(instances.clone()));
        rpass.set_vertex_buffer(PositionInput::VERTEX_ATTRIBUTES[0].shader_location, position.slice(instances.clone()));
        rpass.set_vertex_buffer(SizeInput::VERTEX_ATTRIBUTES[0].shader_location, size.slice(instances.clone()));
        rpass.set_vertex_buffer(ColorInput::VERTEX_ATTRIBUTES[0].shader_location, color.slice(instances.clone()));
        rpass.set_vertex_buffer(ShapeInput::VERTEX_ATTRIBUTES[0].shader_location, shape.slice(instances.clone()));

        let start = u32::try_from(instances.start).unwrap();
        let end = u32::try_from(instances.end).unwrap();
        rpass.draw(0..6, start..end);
    }
}
