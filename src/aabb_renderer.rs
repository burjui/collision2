use std::mem::offset_of;

use wgpu::{
    BlendState, BufferUsages, ColorTargetState, CommandEncoder, Device, MultisampleState, PipelineCache,
    PrimitiveState, Queue, RenderPass, RenderPipeline, RenderPipelineDescriptor, TextureFormat, wgt::DrawIndirectArgs,
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
    node_count: GpuBuffer<u32>,
    draw_args: GpuBuffer<DrawIndirectArgs>,
}

impl AabbRenderer {
    pub fn new(
        device: &Device,
        swapchain_format: TextureFormat,
        pipeline_cache: &PipelineCache,
        camera_buffer: GpuBuffer<Camera>,
        flags: GpuBuffer<Flags>,
        aabbs: GpuBuffer<AABB>,
        node_count: GpuBuffer<u32>,
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
        let bind_group = aabb_frame::WgpuBindGroup0::from_bindings(
            device,
            aabb_frame::WgpuBindGroup0Entries::new(aabb_frame::WgpuBindGroup0EntriesParams {
                camera: camera_buffer.buffer().as_entire_buffer_binding(),
                flags: flags.buffer().as_entire_buffer_binding(),
                aabbs: aabbs.buffer().as_entire_buffer_binding(),
            }),
        );
        let draw_args = GpuBuffer::new(1, "draw args buffer", BufferUsages::INDIRECT | BufferUsages::COPY_DST, device);

        Self {
            render_pipeline,
            bind_group,
            node_count,
            draw_args,
        }
    }

    pub fn prepare(&self, encoder: &mut CommandEncoder, queue: &Queue) {
        let draw_args = DrawIndirectArgs {
            vertex_count: 6,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        };
        self.draw_args.write(queue, &[draw_args]);

        let instant_count_offset = u64::try_from(offset_of!(DrawIndirectArgs, instance_count)).unwrap();
        let instant_count_size = u64::try_from(std::mem::size_of_val::<u32>(&draw_args.instance_count)).unwrap();
        encoder.copy_buffer_to_buffer(
            self.node_count.buffer(),
            0,
            self.draw_args.buffer(),
            instant_count_offset,
            instant_count_size,
        );
    }

    pub fn render(&self, render_pass: &mut RenderPass<'_>) {
        render_pass.set_pipeline(&self.render_pipeline);
        self.bind_group.set(render_pass);
        render_pass.draw_indirect(self.draw_args.buffer(), 0);
    }
}
