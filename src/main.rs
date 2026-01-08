#![allow(clippy::too_many_arguments)]

pub mod aabb;
pub mod aabb_renderer;
pub mod bvh;
pub mod gpu_buffer;
pub mod integration;
pub mod objects;
pub mod scene;
pub mod shaders;
pub mod shape_renderer;

use std::{
    ops::Range,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use crossbeam::channel::Sender;
use pollster::block_on;
use wgpu::{BufferUsages, PollError, PollStatus, PollType, PresentMode, SubmissionIndex};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

use crate::{
    aabb::AabbExt as _,
    aabb_renderer::AabbRenderer,
    gpu_buffer::GpuBuffer,
    integration::GpuIntegrator,
    objects::Objects,
    scene::create_scene,
    shaders::common::{AABB, Camera},
    shape_renderer::ShapeRenderer,
};

fn main() {
    let event_loop = EventLoop::with_user_event().build().expect("Failed to create event loop");
    let event_loop_proxy = event_loop.create_proxy();
    let mut app = App::new(event_loop_proxy);
    event_loop.run_app(&mut app).expect("Failed to run app");
}

struct App<'a> {
    state: Option<AppState<'a>>,
    event_loop_proxy: EventLoopProxy<AppEvent>,
}

impl App<'_> {
    fn new(event_loop_proxy: EventLoopProxy<AppEvent>) -> Self {
        Self {
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
    shape_renderer: ShapeRenderer,
    aabb_renderer: AabbRenderer,
    exit_notification_sender: Sender<()>,
    world_aabb: AABB,
    object_count: usize,
    camera_buffer: GpuBuffer<Camera>,

    surface_config: wgpu::SurfaceConfiguration,
    queue: wgpu::Queue,
    device: wgpu::Device,
    surface: wgpu::Surface<'a>,
    window: Arc<Window>,
}

impl ApplicationHandler<AppEvent> for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut window_attributes = WindowAttributes::default();
        window_attributes.inner_size = Some(PhysicalSize::new(1600, 800).into());
        window_attributes.resizable = true;
        let window = Arc::new(event_loop.create_window(window_attributes).expect("Failed to create window"));
        let wgpu = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let surface = wgpu.create_surface(window.clone()).unwrap();
        let (adapter, device, queue, swapchain_format) = init_wgpu(&wgpu, &surface);
        let window_size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            present_mode: PresentMode::AutoVsync,
            ..surface.get_default_config(&adapter, window_size.width, window_size.height).unwrap()
        };
        surface.configure(&device, &surface_config);

        let world_aabb = AABB {
            min: [-1000.0, -1000.0],
            max: [1000.0, 1000.0],
        };

        let mut objects = Objects::default();
        create_scene(&mut objects, world_aabb);
        let object_count = objects.len();
        let buffers = objects.to_buffers(&device, &queue);

        println!("Window size: {}x{}", window_size.width, window_size.height);
        println!("Object count: {}", object_count);

        let pipeline_cache = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: None,
                data: None,
                fallback: true,
            })
        };
        let camera_buffer =
            GpuBuffer::<Camera>::new(1, "view size buffer", BufferUsages::UNIFORM | BufferUsages::COPY_DST, &device);
        let aabb_renderer = AabbRenderer::new(
            &device,
            swapchain_format,
            &pipeline_cache,
            camera_buffer.clone(),
            buffers.flags.clone(),
            buffers.aabbs.clone(),
        );
        let shape_renderer = ShapeRenderer::new(
            &device,
            swapchain_format,
            &pipeline_cache,
            camera_buffer.clone(),
            buffers.flags.clone(),
            buffers.aabbs.clone(),
            buffers.colors,
            buffers.shapes,
        );
        let (exit_notification_sender, exit_notification_receiver) = crossbeam::channel::bounded(1);

        spawn_simulation_thread(
            buffers.flags,
            buffers.aabbs,
            buffers.velocities,
            buffers.masses,
            device.clone(),
            queue.clone(),
            self.event_loop_proxy.clone(),
            exit_notification_receiver.clone(),
        );

        self.state = Some(AppState {
            shape_renderer,
            aabb_renderer,
            exit_notification_sender,
            world_aabb,
            object_count,
            camera_buffer,

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
                if let Some(state) = &mut self.state {
                    state.surface_config.width = size.width;
                    state.surface_config.height = size.height;
                    state.surface.configure(&state.device, &state.surface_config);
                    state.window.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    let start = Instant::now();

                    let view_size = state.window.inner_size();
                    let world_height = state.world_aabb.max().y - state.world_aabb.min().y;
                    let camera = ortho_camera(view_size.cast(), world_height);
                    state.camera_buffer.write(&state.queue, &[Camera::new(camera)]);

                    let frame = render_scene(
                        &mut state.shape_renderer,
                        &mut state.aabb_renderer,
                        0..state.object_count,
                        &state.device,
                        &state.queue,
                        &state.surface,
                    );

                    state.window.pre_present_notify();
                    frame.present();

                    println!("Rendered {} shapes in {} ms", state.object_count, start.elapsed().as_millis());
                }
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
            let _ = state.exit_notification_sender.try_send(());
        }
    }
}

fn init_wgpu(
    wgpu: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue, wgpu::TextureFormat) {
    let adapter = block_on(wgpu.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::None),
        force_fallback_adapter: false,
        compatible_surface: Some(surface),
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
    (adapter, device, queue, swapchain_format)
}

fn render_scene(
    shape_renderer: &mut ShapeRenderer,
    aabb_renderer: &mut AabbRenderer,
    range: Range<usize>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    surface: &wgpu::Surface,
) -> wgpu::SurfaceTexture {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let frame = surface.get_current_texture().expect("Failed to acquire next swap chain texture");
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

    shape_renderer.render(&mut render_pass, range.clone());
    aabb_renderer.render(&mut render_pass, range);
    drop(render_pass);

    let submission_index = queue.submit(Some(encoder.finish()));
    device.wait_for_submission(submission_index).unwrap();
    frame
}

fn ortho_camera(view_size: PhysicalSize<f32>, world_height: f32) -> [[f32; 4]; 4] {
    let aspect = view_size.width / view_size.height;
    let world_width = world_height * aspect;
    let l = -world_width * 0.5;
    let r = world_width * 0.5;
    let b = -world_height * 0.5;
    let t = world_height * 0.5;
    let sx = 2.0 / (r - l);
    let sy = 2.0 / (t - b);
    let tx = -(r + l) / (r - l);
    let ty = -(t + b) / (t - b);
    [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [tx, ty, 0.0, 1.0],
    ]
}

fn spawn_simulation_thread(
    flags: GpuBuffer<shaders::common::Flags>,
    aabbs: GpuBuffer<shaders::common::AABB>,
    velocities: GpuBuffer<shaders::common::Velocity>,
    masses: GpuBuffer<shaders::common::Mass>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    event_loop_proxy: EventLoopProxy<AppEvent>,
    exit_notification_receiver: crossbeam::channel::Receiver<()>,
) {
    thread::spawn(move || {
        let mut last_redraw = Instant::now();
        let integrator = GpuIntegrator::new(&device);
        let dt_buffer = GpuBuffer::new(1, "dt buffer", BufferUsages::UNIFORM | BufferUsages::COPY_DST, &device);
        dt_buffer.write(&queue, &[0.0001]);
        let processed = GpuBuffer::new(
            1,
            "processed buffer",
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            &device,
        );
        let processed_mapped = GpuBuffer::<u32>::new(
            1,
            "processed mapped buffer",
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            &device,
        );

        loop {
            if exit_notification_receiver.try_recv().is_ok() {
                break;
            }

            let now = Instant::now();
            if now - last_redraw >= Duration::from_secs_f32(1.0 / 60.0) {
                last_redraw = now;
                event_loop_proxy.send_event(AppEvent::RedrawRequested).unwrap();
            }

            processed.write(&queue, &[0]);
            queue.submit([]);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let start = Instant::now();
            integrator.compute(&device, &queue, &dt_buffer, &flags, &aabbs, &velocities, &masses, &processed);
            encoder.copy_buffer_to_buffer(
                processed.buffer(),
                0,
                processed_mapped.buffer(),
                0,
                processed_mapped.buffer().size(),
            );
            let submission_index = queue.submit(Some(encoder.finish()));
            device.wait_for_submission(submission_index).unwrap();

            let processed_value = &mut [0u32];
            processed_mapped.read(&device, processed_value);
            assert_eq!(processed_value[0], u32::try_from(flags.len()).unwrap());

            println!("Integrated {} objects in {} ms", processed_value[0], start.elapsed().as_millis());
        }
    });
}

trait DeviceUtis {
    fn wait_for_submission(&self, submission_index: SubmissionIndex) -> Result<PollStatus, PollError>;
}

impl DeviceUtis for wgpu::Device {
    fn wait_for_submission(&self, submission_index: SubmissionIndex) -> Result<PollStatus, PollError> {
        self.poll(PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        })
    }
}
