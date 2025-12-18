use std::ops::Range;

use wgpu::{
    BlendState, BufferUsages, ColorTargetState, Device, MultisampleState, PipelineCache, PrimitiveState, Queue,
    RenderPass, RenderPipeline, RenderPipelineDescriptor, TextureFormat,
};
use winit::dpi::PhysicalSize;

use crate::{
    gpu_buffer::GpuBuffer,
    shaders::{
        common::{Color, Flags, Position, Shape, Size},
        shape,
    },
};

pub struct ShapeRenderer {
    render_pipeline: RenderPipeline,
    view_size_buffer: GpuBuffer<[f32; 2]>,
    bind_group: shape::WgpuBindGroup0,
}

impl ShapeRenderer {
    pub fn new(
        device: &Device,
        swapchain_format: TextureFormat,
        pipeline_cache: &PipelineCache,
        flags: GpuBuffer<Flags>,
        positions: GpuBuffer<Position>,
        sizes: GpuBuffer<Size>,
        colors: GpuBuffer<Color>,
        shapes: GpuBuffer<Shape>,
    ) -> Self {
        let pipeline_layout = shape::create_pipeline_layout(device);
        let shader = shape::create_shader_module_embed_source(device);

        let vertex_entry = shape::vs_main_entry();
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

        let view_size_buffer =
            GpuBuffer::new(1, "view size buffer", BufferUsages::UNIFORM | BufferUsages::COPY_DST, device);

        let bind_group = shape::WgpuBindGroup0::from_bindings(
            device,
            shape::WgpuBindGroup0Entries::new(shape::WgpuBindGroup0EntriesParams {
                view_size: view_size_buffer.buffer().as_entire_buffer_binding(),
                flags: flags.buffer().as_entire_buffer_binding(),
                position: positions.buffer().as_entire_buffer_binding(),
                size: sizes.buffer().as_entire_buffer_binding(),
                color: colors.buffer().as_entire_buffer_binding(),
                shape: shapes.buffer().as_entire_buffer_binding(),
            }),
        );

        Self {
            render_pipeline,
            view_size_buffer,
            bind_group,
        }
    }

    pub fn prepare(&mut self, queue: &Queue, viewport_size: PhysicalSize<u32>) {
        self.view_size_buffer.write(queue, &[viewport_size.into()]);
    }

    pub fn render(&self, render_pass: &mut RenderPass<'_>, instances: Range<usize>) {
        render_pass.set_pipeline(&self.render_pipeline);
        self.bind_group.set(render_pass);
        let start = u32::try_from(instances.start).unwrap();
        let end = u32::try_from(instances.end).unwrap();
        render_pass.draw(0..6, start..end);
    }
}
