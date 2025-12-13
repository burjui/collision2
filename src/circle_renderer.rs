use nalgebra::{Matrix4, Vector3};
use wgpu::{
    BlendState, Buffer, BufferBinding, BufferUsages, ColorTargetState, Device, MultisampleState, PipelineCache,
    PrimitiveState, RenderPass, RenderPipeline, RenderPipelineDescriptor, TextureFormat, VertexStepMode,
    util::{BufferInitDescriptor, DeviceExt as _},
};
use winit::dpi::PhysicalSize;

use crate::shaders::circle::{self, InstanceInput};

pub struct CircleRenderer {
    vertex_buffer: Buffer,
    instance_buffer: Buffer,
    render_pipeline: RenderPipeline,
    uniforms_bind_group: circle::WgpuBindGroup0,
}

impl CircleRenderer {
    pub fn new(
        device: &Device,
        window_size: PhysicalSize<u32>,
        swapchain_format: TextureFormat,
        pipeline_cache: &PipelineCache,
    ) -> Self {
        let pipeline_layout = circle::create_pipeline_layout(device);
        let shader = circle::create_shader_module_embed_source(device);

        let vertex_entry = circle::vs_main_entry(VertexStepMode::Vertex, VertexStepMode::Instance);
        let vertex_state = circle::vertex_state(&shader, &vertex_entry);

        let color_target_state = ColorTargetState {
            blend: Some(BlendState::ALPHA_BLENDING),
            ..ColorTargetState::from(swapchain_format)
        };
        let fragment_entry = circle::fs_main_entry([Some(color_target_state)]);
        let fragment_state = circle::fragment_state(&shader, &fragment_entry);

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

        let window_size = window_size.cast::<f32>();
        let aspect_ratio = window_size.width / window_size.height;
        let scaling = 0.05;
        let transform_matrix =
            Matrix4::new_nonuniform_scaling(&Vector3::new(1.0 / aspect_ratio, 1.0, 1.0)).append_scaling(scaling);

        let uniforms = circle::Uniforms::new(transform_matrix.into(), scaling);
        let uniforms_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Transform buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let uniforms_bind_group = circle::WgpuBindGroup0::from_bindings(
            device,
            circle::WgpuBindGroup0Entries::new(circle::WgpuBindGroup0EntriesParams {
                uniforms: BufferBinding {
                    buffer: &uniforms_buffer,
                    offset: 0,
                    size: None,
                },
            }),
        );

        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Circle vertex buffer"),
            contents: bytemuck::cast_slice(&[
                circle::VertexInput::new([1.0, 1.0]),
                circle::VertexInput::new([-1.0, 1.0]),
                circle::VertexInput::new([-1.0, -1.0]),
                circle::VertexInput::new([-1.0, -1.0]),
                circle::VertexInput::new([1.0, -1.0]),
                circle::VertexInput::new([1.0, 1.0]),
            ]),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });
        let instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Circle vertex buffer"),
            contents: bytemuck::cast_slice(&[
                InstanceInput::new([0.0, 0.0], 10.0, [1.0, 0.0, 0.0, 1.0]),
                InstanceInput::new([0.0, 2.0], 10.0, [0.0, 1.0, 0.0, 1.0]),
                InstanceInput::new([0.0, 4.0], 10.0, [0.0, 0.0, 1.0, 1.0]),
            ]),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        Self {
            vertex_buffer,
            instance_buffer,
            render_pipeline,
            uniforms_bind_group,
        }
    }

    pub fn prepare(&mut self, device: &Device, instances: &[InstanceInput]) {
        let instance_buffer_size = usize::try_from(self.instance_buffer.size()).unwrap();
        if instance_buffer_size < size_of_val(instances) {
            self.instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Circle vertex buffer"),
                contents: &[],
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            });
        }
    }

    pub fn render(&self, render_pass: &mut RenderPass<'_>) {
        render_pass.set_pipeline(&self.render_pipeline);
        self.uniforms_bind_group.set(render_pass);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.draw(0..6, 0..3);
    }
}
