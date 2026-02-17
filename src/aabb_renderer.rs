use std::array::from_fn;

use wgpu::{
    BlendState, ColorTargetState, Device, MultisampleState, PipelineCache, PrimitiveState, RenderPass, RenderPipeline,
    RenderPipelineDescriptor, TextureFormat,
};

use crate::{
    gpu_buffer::GpuBuffer,
    phase_state::{PhaseState, PhaseStateRing},
    shaders::{
        aabb_frame::{
            self, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1,
            WgpuBindGroup1Entries, WgpuBindGroup1EntriesParams,
        },
        common::Camera,
    },
};

pub struct AabbRenderer {
    node_count: u32,
    render_pipeline: RenderPipeline,
    fixed_bind_group: WgpuBindGroup0,
    phase_state_bind_groups: [Option<WgpuBindGroup1>; PhaseStateRing::CAPACITY],
    phase_state_index: Option<usize>,
}

impl AabbRenderer {
    pub fn new(
        device: &Device,
        swapchain_format: TextureFormat,
        pipeline_cache: &PipelineCache,
        camera_buffer: GpuBuffer<Camera>,
        node_count: u32,
    ) -> Self {
        let pipeline_layout = aabb_frame::create_pipeline_layout(device);
        let shader = aabb_frame::create_shader_module_embed_source(device);
        let vertex_entry = aabb_frame::vs_main_entry();
        let vertex_state = aabb_frame::vertex_state(&shader, &vertex_entry);
        let color_target_state = ColorTargetState {
            blend: Some(BlendState::ALPHA_BLENDING),
            ..ColorTargetState::from(swapchain_format)
        };
        let fragment_entry = aabb_frame::fs_main_entry([Some(color_target_state)]);
        let fragment_state = aabb_frame::fragment_state(&shader, &fragment_entry);
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: vertex_state,
            fragment: Some(fragment_state),
            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                polygon_mode: wgpu::PolygonMode::Line,
                ..PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: Some(pipeline_cache),
        });
        let fixed_bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                camera: camera_buffer.buffer().as_entire_buffer_binding(),
            }),
        );
        Self {
            node_count,
            render_pipeline,
            fixed_bind_group,
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
                }),
            )
        });
        self.phase_state_index = Some(phase_state_index);
    }

    pub fn render(&self, render_pass: &mut RenderPass<'_>) {
        let phase_state_index = self.phase_state_index.expect("prepare() must be called every frame");
        let phase_state_bind_group = self.phase_state_bind_groups[phase_state_index].as_ref().unwrap();
        render_pass.set_pipeline(&self.render_pipeline);
        self.fixed_bind_group.set(render_pass);
        phase_state_bind_group.set(render_pass);
        render_pass.draw(0..6, 0..self.node_count);
    }
}
