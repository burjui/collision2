use std::{array::from_fn, ops::Range};

use wgpu::{
    BlendState, ColorTargetState, Device, MultisampleState, PrimitiveState, RenderPass, RenderPipeline,
    RenderPipelineDescriptor, TextureFormat,
};

use crate::{
    gpu_buffer::GpuBuffer,
    phase_state::{PhaseState, PhaseStateRing},
    shaders::{
        common::{Camera, Color, Shape},
        shape::{
            self, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1,
            WgpuBindGroup1Entries, WgpuBindGroup1EntriesParams,
        },
    },
};

pub struct ShapeRenderer {
    pipeline: RenderPipeline,
    main_bind_group: WgpuBindGroup0,
    phase_state_bind_groups: [Option<WgpuBindGroup1>; PhaseStateRing::CAPACITY],
    phase_state_index: Option<usize>,
}

impl ShapeRenderer {
    pub fn new(
        device: &Device,
        swapchain_format: TextureFormat,
        camera: GpuBuffer<Camera>,
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
            cache: None,
        });
        let main_bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                camera: camera.buffer().as_entire_buffer_binding(),
                colors: colors.buffer().as_entire_buffer_binding(),
                shapes: shapes.buffer().as_entire_buffer_binding(),
            }),
        );
        Self {
            pipeline: render_pipeline,
            main_bind_group,
            phase_state_bind_groups: from_fn(|_| None),
            phase_state_index: None,
        }
    }

    pub fn prepare(&mut self, phase_state_index: usize, device: &Device, phase_state: &PhaseState) {
        self.phase_state_bind_groups[phase_state_index].get_or_insert_with(|| {
            WgpuBindGroup1::from_bindings(
                device,
                WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                    flags: phase_state.flags().buffer().as_entire_buffer_binding(),
                    aabbs: phase_state.aabbs().buffer().as_entire_buffer_binding(),
                    velocities: phase_state.velocities().buffer().as_entire_buffer_binding(),
                }),
            )
        });
        self.phase_state_index = Some(phase_state_index);
    }

    pub fn render(&self, render_pass: &mut RenderPass<'_>, instances: Range<usize>) {
        let phase_state_index = self.phase_state_index.expect("prepare() must be called every frame");
        let phase_state_bind_group = self.phase_state_bind_groups[phase_state_index].as_ref().unwrap();
        let start = u32::try_from(instances.start).unwrap();
        let end = u32::try_from(instances.end).unwrap();
        render_pass.set_pipeline(&self.pipeline);
        self.main_bind_group.set(render_pass);
        phase_state_bind_group.set(render_pass);
        render_pass.draw(0..6, start..end);
    }
}
