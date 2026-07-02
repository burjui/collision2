use std::ops::Range;

use wgpu::{
    BlendState, ColorTargetState, Device, MultisampleState, PrimitiveState, RenderPass, RenderPipeline,
    RenderPipelineDescriptor, TextureFormat,
};

use crate::{
    device_buffer::DeviceBuffer,
    phase_state::PhaseState,
    shaders::{
        common::{Camera, Color, Mass, Shape},
        render_shape::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1, WgpuBindGroup1Entries,
            WgpuBindGroup1EntriesParams, create_pipeline_layout, create_shader_module_embed_source, fragment_state,
            fs_main_entry, vertex_state, vs_main_entry,
        },
    },
    util::PhaseStateCache,
};

pub struct ShapeRenderer {
    pipeline: RenderPipeline,
    main_bind_group: WgpuBindGroup0,
    phase_state_cache: PhaseStateCache<WgpuBindGroup1>,
}

impl ShapeRenderer {
    pub fn new(
        device: &Device,
        swapchain_format: TextureFormat,
        camera: DeviceBuffer<Camera>,
        colors: DeviceBuffer<Color>,
        shapes: DeviceBuffer<Shape>,
        masses: DeviceBuffer<Mass>,
    ) -> Self {
        let pipeline_layout = create_pipeline_layout(device);
        let shader = create_shader_module_embed_source(device);
        let vertex_entry = vs_main_entry();
        let vertex_state = vertex_state(&shader, &vertex_entry);
        let color_target_state = ColorTargetState {
            blend: Some(BlendState::ALPHA_BLENDING),
            ..ColorTargetState::from(swapchain_format)
        };
        let fragment_entry = fs_main_entry([Some(color_target_state)]);
        let fragment_state = fragment_state(&shader, &fragment_entry);
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: vertex_state,
            fragment: Some(fragment_state),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });
        let main_bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                camera: camera.as_entire_buffer_binding(),
                colors: colors.as_entire_buffer_binding(),
                shapes: shapes.as_entire_buffer_binding(),
                masses: masses.as_entire_buffer_binding(),
            }),
        );
        let phase_state_cache = PhaseStateCache::new();
        Self {
            pipeline: render_pipeline,
            main_bind_group,
            phase_state_cache,
        }
    }

    pub fn prepare(&mut self, phase_state_index: usize, device: &Device, phase_state: &PhaseState) {
        self.phase_state_cache.update(phase_state_index, || {
            WgpuBindGroup1::from_bindings(
                device,
                WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                    flags: phase_state.flags().as_entire_buffer_binding(),
                    aabbs: phase_state.aabbs().as_entire_buffer_binding(),
                    velocities: phase_state.velocities().as_entire_buffer_binding(),
                }),
            )
        });
    }

    pub fn render(&self, render_pass: &mut RenderPass<'_>, instances: Range<usize>) {
        let phase_state_bind_group = self.phase_state_cache.get_current();
        let start = u32::try_from(instances.start).unwrap();
        let end = u32::try_from(instances.end).unwrap();
        render_pass.set_pipeline(&self.pipeline);
        self.main_bind_group.set(render_pass);
        phase_state_bind_group.set(render_pass);
        render_pass.draw(0..6, start..end);
    }
}
