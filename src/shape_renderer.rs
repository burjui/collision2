use std::ops::Range;

use wgpu::{
    BlendState, BufferUsages, ColorTargetState, Device, MultisampleState, PipelineCache, PrimitiveState, Queue,
    RenderPass, RenderPipeline, RenderPipelineDescriptor, TextureFormat, VertexStepMode,
};
use winit::dpi::PhysicalSize;

use crate::{
    gpu_buffer::GpuBuffer,
    shaders::shape::{self, ColorInput, FlagsInput, PositionInput, ShapeInput, SizeInput, VertexInput},
};

pub struct ShapeRenderer {
    vertex_buffer: GpuBuffer<VertexInput>,
    render_pipeline: RenderPipeline,
    uniforms_buffer: GpuBuffer<shape::Uniforms>,
    uniforms_bind_group: shape::WgpuBindGroup0,
}

impl ShapeRenderer {
    pub fn new(
        device: &Device,
        queue: &Queue,
        swapchain_format: TextureFormat,
        pipeline_cache: &PipelineCache,
    ) -> Self {
        let vertex_buffer =
            GpuBuffer::new(6, "Shape vertex buffer", BufferUsages::VERTEX | BufferUsages::COPY_DST, device);
        vertex_buffer.write(
            queue,
            &[
                VertexInput::new([1.0, 1.0]),
                VertexInput::new([-1.0, 1.0]),
                VertexInput::new([-1.0, -1.0]),
                VertexInput::new([-1.0, -1.0]),
                VertexInput::new([1.0, -1.0]),
                VertexInput::new([1.0, 1.0]),
            ],
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
            GpuBuffer::new(1, "Shape uniforms buffer", BufferUsages::UNIFORM | BufferUsages::COPY_DST, device);
        let uniforms_bind_group = shape::WgpuBindGroup0::from_bindings(
            device,
            shape::WgpuBindGroup0Entries::new(shape::WgpuBindGroup0EntriesParams {
                uniforms: uniforms_buffer.buffer().as_entire_buffer_binding(),
            }),
        );

        Self {
            vertex_buffer,
            render_pipeline,
            uniforms_buffer,
            uniforms_bind_group,
        }
    }

    pub fn prepare(&mut self, queue: &Queue, viewport_size: PhysicalSize<u32>) {
        let uniforms = [shape::Uniforms::new(viewport_size.into())];
        self.uniforms_buffer.write(queue, &uniforms);
    }

    pub fn render(
        &self,
        rpass: &mut RenderPass<'_>,
        instances: Range<usize>,
        flags: &GpuBuffer<shape::FlagsInput>,
        position: &GpuBuffer<shape::PositionInput>,
        size: &GpuBuffer<shape::SizeInput>,
        color: &GpuBuffer<shape::ColorInput>,
        shape: &GpuBuffer<shape::ShapeInput>,
    ) {
        rpass.set_pipeline(&self.render_pipeline);
        self.uniforms_bind_group.set(rpass);

        rpass.set_vertex_buffer(VertexInput::VERTEX_ATTRIBUTES[0].shader_location, self.vertex_buffer.slice(..));
        rpass.set_vertex_buffer(FlagsInput::VERTEX_ATTRIBUTES[0].shader_location, flags.slice(..));
        rpass.set_vertex_buffer(PositionInput::VERTEX_ATTRIBUTES[0].shader_location, position.slice(..));
        rpass.set_vertex_buffer(SizeInput::VERTEX_ATTRIBUTES[0].shader_location, size.slice(..));
        rpass.set_vertex_buffer(ColorInput::VERTEX_ATTRIBUTES[0].shader_location, color.slice(..));
        rpass.set_vertex_buffer(ShapeInput::VERTEX_ATTRIBUTES[0].shader_location, shape.slice(..));

        let start = u32::try_from(instances.start).unwrap();
        let end = u32::try_from(instances.end).unwrap();
        rpass.draw(0..6, start..end);
    }
}
