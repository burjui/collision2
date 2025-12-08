use pollster::block_on;
use std::{borrow::Cow, sync::Arc};
use wgpu::ColorTargetState;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes, WindowId},
};

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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];
        let color_target_state = ColorTargetState {
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            ..wgpu::ColorTargetState::from(swapchain_format)
        };
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
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(color_target_state)],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: Some(&pipeline_cache),
        });

        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        surface.configure(&device, &config);

        self.state = Some(AppState {
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
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                    rpass.set_pipeline(&state.render_pipeline);
                    rpass.draw(0..6, 0..2);
                }

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
