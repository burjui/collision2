use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

use wgpu::{
    BlendState, ColorTargetState, Device, MultisampleState, PipelineCache, PrimitiveState, RenderPass, RenderPipeline,
    RenderPipelineDescriptor, TextureFormat,
};

use crate::{
    gpu_buffer::GpuBuffer,
    phase_state::PhaseStateBuffers,
    shaders::{
        common::{Camera, Color, Flags, Shape},
        shape,
    },
};

pub struct ShapeRenderer {
    device: Device,
    camera: GpuBuffer<Camera>,
    size_factor: GpuBuffer<f32>,
    flags: GpuBuffer<Flags>,
    colors: GpuBuffer<Color>,
    shapes: GpuBuffer<Shape>,
    phase_state_buffers: Arc<Mutex<PhaseStateBuffers>>,
    render_pipeline: RenderPipeline,
}

impl ShapeRenderer {
    pub fn new(
        device: Device,
        swapchain_format: TextureFormat,
        pipeline_cache: &PipelineCache,
        camera: GpuBuffer<Camera>,
        size_factor: GpuBuffer<f32>,
        flags: GpuBuffer<Flags>,
        colors: GpuBuffer<Color>,
        shapes: GpuBuffer<Shape>,
        phase_state_buffers: Arc<Mutex<PhaseStateBuffers>>,
    ) -> Self {
        let pipeline_layout = shape::create_pipeline_layout(&device);
        let shader = shape::create_shader_module_embed_source(&device);

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

        Self {
            device,
            camera,
            size_factor,
            flags,
            colors,
            shapes,
            phase_state_buffers,
            render_pipeline,
        }
    }

    pub fn render(&self, render_pass: &mut RenderPass<'_>, instances: Range<usize>) {
        let guard = self.phase_state_buffers.lock().unwrap();
        let phase_states = guard.oldest();
        drop(guard);
        let bind_group = shape::WgpuBindGroup0::from_bindings(
            &self.device,
            shape::WgpuBindGroup0Entries::new(shape::WgpuBindGroup0EntriesParams {
                camera: self.camera.buffer().as_entire_buffer_binding(),
                size_factor: self.size_factor.buffer().as_entire_buffer_binding(),
                flags: self.flags.buffer().as_entire_buffer_binding(),
                aabbs: phase_states.aabbs().buffer().as_entire_buffer_binding(),
                colors: self.colors.buffer().as_entire_buffer_binding(),
                shapes: self.shapes.buffer().as_entire_buffer_binding(),
                velocities: phase_states.velocities().buffer().as_entire_buffer_binding(),
            }),
        );

        render_pass.set_pipeline(&self.render_pipeline);
        bind_group.set(render_pass);
        let start = u32::try_from(instances.start).unwrap();
        let end = u32::try_from(instances.end).unwrap();
        render_pass.draw(0..6, start..end);
    }
}
