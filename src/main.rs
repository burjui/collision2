#![allow(clippy::too_many_arguments)]

pub mod gpu_arena;
pub mod integration;
pub mod objects;
pub mod shaders;
pub mod shape_renderer;

use core::f32;
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use crossbeam::channel::Sender;
use itertools::Itertools;
use nalgebra::Vector2;
use pollster::block_on;
use rand::random;
use wgpu::{BufferUsages, PollType};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

use crate::{
    gpu_arena::{GpuArena, GpuSlice},
    integration::GpuIntegrator,
    objects::{ObjectPrototype, Objects},
    shaders::{
        common::{FLAG_PHYSICAL, FLAG_SHOW},
        shape::{self},
    },
    shape_renderer::ShapeRenderer,
};

fn main() {
    let event_loop = EventLoop::with_user_event().build().expect("Failed to create event loop");
    let event_loop_proxy = event_loop.create_proxy();
    let mut app = App::new(event_loop_proxy);
    event_loop.run_app(&mut app).expect("Failed to run app");
}

struct App<'a> {
    wgpu: wgpu::Instance,
    state: Option<AppState<'a>>,
    event_loop_proxy: EventLoopProxy<AppEvent>,
}

impl App<'_> {
    fn new(event_loop_proxy: EventLoopProxy<AppEvent>) -> Self {
        Self {
            wgpu: wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default()),
            state: None,
            event_loop_proxy,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum AppEvent {
    RedrawRequested,
}

struct AppState<'a> {
    flags: GpuSlice<shape::FlagsInput>,
    positions: GpuSlice<shape::PositionInput>,
    sizes: GpuSlice<shape::SizeInput>,
    colors: GpuSlice<shape::ColorInput>,
    shapes: GpuSlice<shape::ShapeInput>,

    shape_renderer: ShapeRenderer,
    exit_notification_sender: Sender<()>,

    surface_config: wgpu::SurfaceConfiguration,
    queue: wgpu::Queue,
    device: wgpu::Device,
    surface: wgpu::Surface<'a>,
    window: Arc<Window>,
}

impl ApplicationHandler<AppEvent> for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // TODO refactor this pile

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

        println!("Max bind groups: {}", device.limits().max_bind_groups);
        println!("Max bindings per bind group: {}", device.limits().max_bindings_per_bind_group);
        println!("Max uniform buffers per shader stage: {}", device.limits().max_uniform_buffers_per_shader_stage);

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let window_size = window.inner_size();
        let surface_config = surface.get_default_config(&adapter, window_size.width, window_size.height).unwrap();
        surface.configure(&device, &surface_config);

        let window_size = Vector2::<f32>::new(window_size.cast().width, window_size.cast().height);
        println!("Window size: {}x{}", window_size.x, window_size.y);

        let mut objects = Objects::default();
        let circles = {
            const RADIUS: f32 = 0.2;
            // const VELOCITY_MAX: f32 = 0.01;

            let shape_count = window_size * 0.5 / RADIUS;
            println!("Shape count: {}", (shape_count.x * shape_count.y) as usize);
            (0..shape_count.x as usize).cartesian_product(0..shape_count.y as usize).map(move |(i, j)| {
                let (i, j) = (i as f32, j as f32);
                let position = [RADIUS * (i * 2.0 + 1.0), RADIUS * (j * 2.0 + 1.0)];
                ObjectPrototype {
                    mass: 1.0,
                    velocity: [
                        // random_range(-VELOCITY_MAX..VELOCITY_MAX),
                        // random_range(-VELOCITY_MAX..VELOCITY_MAX),
                        0.0, 0.0,
                    ],
                    flags: FLAG_SHOW | FLAG_PHYSICAL,
                    position,
                    size: [RADIUS * 2.0, RADIUS * 2.0],
                    color: [random(), random(), random(), 1.0],
                    shape: shape::SHAPE_CIRCLE,
                }
            })
        };

        const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        let top = ObjectPrototype {
            mass: f32::INFINITY,
            velocity: [0.0, 0.0],
            flags: FLAG_SHOW,
            position: [window_size.x / 2.0, 0.5],
            size: [window_size.x, 1.0],
            color: RED,
            shape: shape::SHAPE_RECT,
        };
        let bottom = ObjectPrototype {
            mass: f32::INFINITY,
            velocity: [0.0, 0.0],
            flags: FLAG_SHOW,
            position: [window_size.x / 2.0, window_size.y - 0.5],
            size: [window_size.x, 1.0],
            color: RED,
            shape: shape::SHAPE_RECT,
        };
        let left = ObjectPrototype {
            mass: f32::INFINITY,
            velocity: [0.0, 0.0],
            flags: FLAG_SHOW,
            position: [0.5, window_size.y / 2.0],
            size: [1.0, window_size.y],
            color: RED,
            shape: shape::SHAPE_RECT,
        };
        let right = ObjectPrototype {
            mass: f32::INFINITY,
            velocity: [0.0, 0.0],
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

        let access_mode = BufferUsages::COPY_DST;
        let (_, velocities) =
            GpuArena::new_slice(objects.len(), "Velocity arena", BufferUsages::STORAGE | access_mode, &device);
        let (_, masses) =
            GpuArena::new_slice(objects.len(), "Mass arena", BufferUsages::STORAGE | access_mode, &device);
        let (_, flags) = GpuArena::new_slice(
            objects.len(),
            "Flags arena",
            BufferUsages::VERTEX | BufferUsages::STORAGE | access_mode,
            &device,
        );
        let (_, positions) = GpuArena::new_slice(
            objects.len(),
            "Position arena",
            BufferUsages::VERTEX | BufferUsages::STORAGE | access_mode,
            &device,
        );
        let (_, sizes) = GpuArena::new_slice(objects.len(), "Size arena", BufferUsages::VERTEX | access_mode, &device);
        let (_, colors) =
            GpuArena::new_slice(objects.len(), "Color arena", BufferUsages::VERTEX | access_mode, &device);
        let (_, shapes) =
            GpuArena::new_slice(objects.len(), "Shape arena", BufferUsages::VERTEX | access_mode, &device);

        masses.write(&queue, &objects.mass);
        velocities.write(&queue, &objects.velocity);
        flags.write(&queue, &objects.flags);
        positions.write(&queue, &objects.position);
        sizes.write(&queue, &objects.size);
        colors.write(&queue, &objects.color);
        shapes.write(&queue, &objects.shape);

        let pipeline_cache = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: None,
                data: None,
                fallback: true,
            })
        };
        let shape_renderer = ShapeRenderer::new(&device, &queue, swapchain_format, &pipeline_cache);
        let (exit_notification_sender, exit_notification_receiver) = crossbeam::channel::bounded(1);

        {
            let window = window.clone();
            let device = device.clone();
            let queue = queue.clone();
            let positions = positions.clone();
            let flags = flags.clone();
            let event_loop_proxy = self.event_loop_proxy.clone();
            let exit_notification_receiver = exit_notification_receiver.clone();
            thread::spawn(move || {
                let mut last_integration = Instant::now();
                let mut last_redraw = Instant::now();
                let integrator = GpuIntegrator::new(&device);
                let (_, dt_buffer) =
                    GpuArena::new_slice(1, "Delta time arena", BufferUsages::STORAGE | access_mode, &device);
                dt_buffer.write(&queue, &[0.001]);

                loop {
                    if exit_notification_receiver.try_recv().is_ok() {
                        break;
                    }

                    let now = Instant::now();
                    if now - last_redraw >= Duration::from_secs_f32(1.0 / 60.0) {
                        last_redraw = now;
                        event_loop_proxy.send_event(AppEvent::RedrawRequested).unwrap();
                        window.request_redraw();
                    }

                    let dt = (now - last_integration).as_secs_f32();
                    last_integration = now;

                    dt_buffer.write(&queue, &[dt]);
                    let submission_index = integrator.compute(
                        &device,
                        &queue,
                        &dt_buffer,
                        &positions.cast(),
                        &velocities,
                        &masses,
                        &flags,
                    );
                    device
                        .poll(PollType::Wait {
                            submission_index: Some(submission_index),
                            timeout: None,
                        })
                        .unwrap();
                }
            });
        }

        self.state = Some(AppState {
            flags,
            positions,
            sizes,
            colors,
            shapes,

            shape_renderer,
            exit_notification_sender,

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

                state.shape_renderer.prepare(&state.queue, state.window.inner_size());
                let object_count = state.flags.len();
                state.shape_renderer.render(
                    &mut render_pass,
                    0..object_count,
                    &state.flags,
                    &state.positions,
                    &state.sizes,
                    &state.colors,
                    &state.shapes,
                );

                drop(render_pass);

                state.queue.submit(Some(encoder.finish()));
                state.window.pre_present_notify();
                frame.present();

                println!("Rendered {} objects in {} ms", object_count, start.elapsed().as_millis());
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

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::RedrawRequested => {
                if let Some(state) = &self.state {
                    state.window.request_redraw();
                }
            }
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.exit_notification_sender.send(()).unwrap();
        }
    }
}
