pub mod objects;
pub mod shape_renderer;
mod shape_shaders;
pub mod wgpu_buffer;

use std::{sync::Arc, time::Instant};

use itertools::Itertools;
use nalgebra::Vector2;
use pollster::block_on;
use rand::random;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

use crate::{
    objects::{ObjectPrototype, Objects},
    shape_renderer::ShapeRenderer,
    shape_shaders::shape::{self, FLAG_SHOW},
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
    objects: Objects,
    shape_renderer: ShapeRenderer,
    surface_config: wgpu::SurfaceConfiguration,
    queue: wgpu::Queue,
    device: wgpu::Device,
    surface: wgpu::Surface<'a>,
    window: Arc<Window>,
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut window_attributes = WindowAttributes::default();
        window_attributes.inner_size = Some(PhysicalSize::new(1600, 800).into());
        window_attributes.resizable = true;
        let window = Arc::new(event_loop.create_window(window_attributes).expect("Failed to create window"));
        let surface = self.wgpu.create_surface(window.clone()).unwrap();

        let adapter = block_on(self.wgpu.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::None),
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

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let window_size = window.inner_size();
        let surface_config = surface.get_default_config(&adapter, window_size.width, window_size.height).unwrap();
        surface.configure(&device, &surface_config);

        let window_size = Vector2::<f32>::new(window_size.cast().width, window_size.cast().height);
        println!("Window size: {}x{}", window_size.x, window_size.y);

        let mut objects = Objects::new();
        let circles = {
            const RADIUS: f32 = 1.0;
            let shape_count = window_size * 0.5 / RADIUS;
            println!("Shape count: {}", (shape_count.x * shape_count.y) as usize);
            (0..shape_count.x as usize).cartesian_product(0..shape_count.y as usize).map(move |(i, j)| {
                let (i, j) = (i as f32, j as f32);
                let position = [RADIUS * (i * 2.0 + 1.0), RADIUS * (j * 2.0 + 1.0)];
                ObjectPrototype {
                    flags: FLAG_SHOW,
                    position,
                    size: [RADIUS * 2.0, RADIUS * 2.0],
                    color: [random(), random(), random(), 1.0],
                    shape: shape::SHAPE_CIRCLE,
                }
            })
        };

        const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        let top = ObjectPrototype {
            flags: FLAG_SHOW,
            position: [window_size.x / 2.0, 0.5],
            size: [window_size.x, 1.0],
            color: RED,
            shape: shape::SHAPE_RECT,
        };
        let bottom = ObjectPrototype {
            flags: FLAG_SHOW,
            position: [window_size.x / 2.0, window_size.y - 0.5],
            size: [window_size.x, 1.0],
            color: RED,
            shape: shape::SHAPE_RECT,
        };
        let left = ObjectPrototype {
            flags: FLAG_SHOW,
            position: [0.5, window_size.y / 2.0],
            size: [1.0, window_size.y],
            color: RED,
            shape: shape::SHAPE_RECT,
        };
        let right = ObjectPrototype {
            flags: FLAG_SHOW,
            position: [window_size.x - 0.5, window_size.y / 2.0],
            size: [1.0, window_size.y],
            color: RED,
            shape: shape::SHAPE_RECT,
        };
        objects.extend(circles);
        objects.push(top);
        objects.push(bottom);
        objects.push(left);
        objects.push(right);

        let pipeline_cache = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: None,
                data: None,
                fallback: true,
            })
        };
        let shape_renderer = ShapeRenderer::new(&device, &queue, swapchain_format, &pipeline_cache);

        self.state = Some(AppState {
            objects,
            shape_renderer,
            surface_config,
            queue,
            device,
            surface,
            window,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::Resized(size) => {
                let state = self.state.as_mut().unwrap();
                state.surface_config.width = size.width;
                state.surface_config.height = size.height;
                state.surface.configure(&state.device, &state.surface_config);
                state.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                let start = Instant::now();

                let state = self.state.as_mut().unwrap();
                let mut encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                let frame = state.surface.get_current_texture().expect("Failed to acquire next swap chain texture");
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
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

                println!("Window inner size: {:?}", state.window.inner_size());
                state.shape_renderer.prepare(&state.device, &state.queue, &state.objects, state.window.inner_size());
                state.shape_renderer.render(&mut render_pass, 0..state.objects.len());

                drop(render_pass);
                let submission_index = state.queue.submit(Some(encoder.finish()));
                state
                    .device
                    .poll(wgpu::PollType::Wait {
                        submission_index: Some(submission_index),
                        timeout: None,
                    })
                    .unwrap();
                state.window.pre_present_notify();
                frame.present();

                println!("Rendered in {} ms", start.elapsed().as_millis());
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyR),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if let Some(state) = &self.state {
                    state.window.request_redraw();
                }
            }

            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),

            _ => (),
        }
    }
}
