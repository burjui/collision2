use std::ops::Range;

use wgpu::{
    BlendState, ColorTargetState, Device, MultisampleState, PipelineCache, PrimitiveState, RenderPass, RenderPipeline,
    RenderPipelineDescriptor, TextureFormat,
};

use crate::{
    gpu_buffer::GpuBuffer,
    shaders::{
        aabb_frame,
        common::{AABB, Camera, Flags},
    },
};

pub struct AabbRenderer {
    render_pipeline: RenderPipeline,
    bind_group: aabb_frame::WgpuBindGroup0,
}

impl AabbRenderer {
    pub fn new(
        device: &Device,
        swapchain_format: TextureFormat,
        pipeline_cache: &PipelineCache,
        camera_buffer: GpuBuffer<Camera>,
        flags: GpuBuffer<Flags>,
        aabbs: GpuBuffer<AABB>,
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
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: Some(pipeline_cache),
        });

        let bind_group = aabb_frame::WgpuBindGroup0::from_bindings(
            device,
            aabb_frame::WgpuBindGroup0Entries::new(aabb_frame::WgpuBindGroup0EntriesParams {
                camera: camera_buffer.buffer().as_entire_buffer_binding(),
                flags: flags.buffer().as_entire_buffer_binding(),
                aabbs: aabbs.buffer().as_entire_buffer_binding(),
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
