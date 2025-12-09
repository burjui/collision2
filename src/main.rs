mod shaders;

use nalgebra::{Matrix4, Vector3};
use pollster::block_on;
use std::sync::Arc;
use wgpu::{util::DeviceExt, ColorTargetState};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes, WindowId},
};

use crate::shaders::circle;

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Failed to run app");
}

struct App<'a> {
    wgpu: wgpu::Instance,
    state: Option<AppState<'a>>,
}

impl App<'_> {
    fn new() -> Self {
        Self {
            wgpu: wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default()),
            state: None,
        }
    }
}

struct AppState<'a> {
    vertex_buffer: wgpu::Buffer,
    uniforms_bind_group: circle::WgpuBindGroup0,
    _uniforms_buffer: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,
    queue: wgpu::Queue,
    device: wgpu::Device,
    adapter: wgpu::Adapter,
    surface: wgpu::Surface<'a>,
    window: Arc<Window>,
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // event_loop.set_control_flow(ControlFlow::Poll);
        let mut window_attributes = WindowAttributes::default();
        window_attributes.inner_size = Some(PhysicalSize::new(800, 600).into());
        window_attributes.resizable = false;
        let window = Arc::new(
            event_loop
                .create_window(window_attributes)
                .expect("Failed to create window"),
        );
        let surface = self.wgpu.create_surface(window.clone()).unwrap();

        let adapter = block_on(self.wgpu.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference:
                wgpu::PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::None),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .expect("Failed to find an appropriate adapter");

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::PIPELINE_CACHE,
            required_limits: wgpu::Limits::defaults().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .expect("Failed to create device");

        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        surface.configure(&device, &config);

        let shader = circle::create_shader_module_embed_source(&device);
        let pipeline_layout = circle::create_pipeline_layout(&device);

        let vertex_entry = circle::vs_main_entry(wgpu::VertexStepMode::Vertex);
        let vertex_state = circle::vertex_state(&shader, &vertex_entry);

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];
        let color_target_state = ColorTargetState {
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            ..wgpu::ColorTargetState::from(swapchain_format)
        };
        let fragment_entry = circle::fs_main_entry([Some(color_target_state)]);
        let fragment_state = circle::fragment_state(&shader, &fragment_entry);

        let pipeline_cache = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: None,
                data: None,
                fallback: true,
            })
        };
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: vertex_state,
            fragment: Some(fragment_state),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: Some(&pipeline_cache),
        });

        let size_f32 = size.cast::<f32>();
        let aspect_ratio = size_f32.width / size_f32.height;
        let transform_matrix =
            Matrix4::new_nonuniform_scaling(&Vector3::new(1.0 / aspect_ratio, 1.0, 1.0));
        let uniforms = circle::Uniforms::new(transform_matrix.into());
        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniforms_bind_group = circle::WgpuBindGroup0::from_bindings(
            &device,
            circle::WgpuBindGroup0Entries::new(circle::WgpuBindGroup0EntriesParams {
                uniforms: wgpu::BufferBinding {
                    buffer: &uniforms_buffer,
                    offset: 0,
                    size: None,
                },
            }),
        );

        let circle_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex buffer"),
            contents: bytemuck::cast_slice(&[
                circle::VertexInput::new([1.0, 1.0]),
                circle::VertexInput::new([-1.0, 1.0]),
                circle::VertexInput::new([-1.0, -1.0]),
                circle::VertexInput::new([-1.0, -1.0]),
                circle::VertexInput::new([1.0, -1.0]),
                circle::VertexInput::new([1.0, 1.0]),
            ]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        self.state = Some(AppState {
            vertex_buffer: circle_vertex_buffer,
            uniforms_bind_group,
            _uniforms_buffer: uniforms_buffer,
            render_pipeline,
            queue,
            device,
            adapter,
            surface,
            window,
        })
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(size) => {
                let state = self.state.as_ref().unwrap();
                let config = state
                    .surface
                    .get_default_config(&state.adapter, size.width, size.height)
                    .unwrap();
                state.surface.configure(&state.device, &config);
                state.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                let state = self.state.as_ref().unwrap();
                let frame = state
                    .surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = state
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                render_pass.set_pipeline(&state.render_pipeline);
                state.uniforms_bind_group.set(&mut render_pass);
                render_pass.set_vertex_buffer(0, state.vertex_buffer.slice(..));
                render_pass.draw(0..6, 0..2);
                drop(render_pass);
                state.queue.submit(Some(encoder.finish()));
                state.window.pre_present_notify();
                frame.present();
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),

            WindowEvent::CloseRequested => event_loop.exit(),

            _ => (),
        }
    }
}
