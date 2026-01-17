use std::ops::Range;

use wgpu::{
    BlendState, ColorTargetState, Device, MultisampleState, PipelineCache, PrimitiveState, RenderPass, RenderPipeline,
    RenderPipelineDescriptor, TextureFormat,
};

use crate::{
    gpu_buffer::GpuBuffer,
    shaders::{
        common::{AABB, Camera, Color, Flags, Shape, Velocity},
        shape,
    },
};

pub struct ShapeRenderer {
    render_pipeline: RenderPipeline,
    bind_group: shape::WgpuBindGroup0,
}

impl ShapeRenderer {
    pub fn new(
        device: &Device,
        swapchain_format: TextureFormat,
        pipeline_cache: &PipelineCache,
        camera: GpuBuffer<Camera>,
        size_factor: GpuBuffer<f32>,
        flags: GpuBuffer<Flags>,
        aabbs: GpuBuffer<AABB>,
        colors: GpuBuffer<Color>,
        shapes: GpuBuffer<Shape>,
        velocities: GpuBuffer<Velocity>,
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

        let bind_group = shape::WgpuBindGroup0::from_bindings(
            device,
            shape::WgpuBindGroup0Entries::new(shape::WgpuBindGroup0EntriesParams {
                camera: camera.buffer().as_entire_buffer_binding(),
                size_factor: size_factor.buffer().as_entire_buffer_binding(),
                flags: flags.buffer().as_entire_buffer_binding(),
                aabbs: aabbs.buffer().as_entire_buffer_binding(),
                colors: colors.buffer().as_entire_buffer_binding(),
                shapes: shapes.buffer().as_entire_buffer_binding(),
                velocities: velocities.buffer().as_entire_buffer_binding(),
            }),
        );

        Self {
            render_pipeline,
            bind_group,
        }
    }

    pub fn render(&self, render_pass: &mut RenderPass<'_>, instances: Range<usize>) {
        render_pass.set_pipeline(&self.render_pipeline);
        self.bind_group.set(render_pass);
        let start = u32::try_from(instances.start).unwrap();
        let end = u32::try_from(instances.end).unwrap();
        render_pass.draw(0..6, start..end);
    }
}
