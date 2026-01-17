#![allow(clippy::too_many_arguments)]

pub mod aabb;
pub mod aabb_renderer;
pub mod bvh_builder;
pub mod gpu_buffer;
pub mod integration;
#[cfg(test)]
mod mock_bvh_test;
pub mod objects;
pub mod pass_duration;
pub mod scene;
pub mod shaders;
pub mod shape_renderer;
#[allow(unused)]
pub mod util;

use crate::{
    aabb_renderer::AabbRenderer,
    bvh_builder::BvhBuilder,
    gpu_buffer::GpuBuffer,
    integration::GpuIntegrator,
    objects::Objects,
    pass_duration::PassDurationMeasurer,
    scene::create_scene,
    shaders::{
        bvh::CombineNodePass,
        common::{AABB, BvhNode, Camera},
    },
    shape_renderer::ShapeRenderer,
    util::DeviceUtil,
};
use crossbeam::channel::Sender;
use pollster::block_on;
use shaders::common::{Flags, Mass, Velocity};
use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use wgpu::{
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, PipelineCacheDescriptor, PresentMode,
    RenderPassColorAttachment, RenderPassDescriptor, RequestAdapterOptions, SubmissionIndex, TextureFormat,
    TextureView, TextureViewDescriptor,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{Fullscreen, Window, WindowAttributes, WindowId},
};

fn main() {
    let event_loop = EventLoop::with_user_event().build().expect("Failed to create event loop");
    let event_loop_proxy = event_loop.create_proxy();
    let mut app = App::new(event_loop_proxy);
    event_loop.run_app(&mut app).expect("Failed to run app");
}

struct App<'a> {
    render_parameters: RenderParameters,
    gpu_state: Option<GpuState<'a>>,
    event_loop_proxy: EventLoopProxy<AppEvent>,
}

impl App<'_> {
    fn new(event_loop_proxy: EventLoopProxy<AppEvent>) -> Self {
        Self {
            render_parameters: RenderParameters::default(),
            gpu_state: None,
            event_loop_proxy,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum AppEvent {
    RedrawRequested,
}

struct RenderParameters {
    enabled: bool,
    draw_aabbs: bool,
    zoom: f32,
}

impl Default for RenderParameters {
    fn default() -> Self {
        Self {
            enabled: true,
            draw_aabbs: false,
            zoom: 1.0,
        }
    }
}

struct GpuState<'a> {
    shape_renderer: ShapeRenderer,
    aabb_renderer: AabbRenderer,
    exit_notification_sender: Sender<()>,
    world_aabb: AABB,
    object_count: usize,
    camera_buffer: GpuBuffer<Camera>,
    node_count_atomic: Arc<AtomicU32>,

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
        window_attributes.fullscreen = Some(Fullscreen::Borderless(None));
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
            device.create_pipeline_cache(&PipelineCacheDescriptor {
                label: None,
                data: None,
                fallback: true,
            })
        };
        let camera_buffer =
            GpuBuffer::<Camera>::new(1, "camera buffer", BufferUsages::UNIFORM | BufferUsages::COPY_DST, &device);
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
        let node_count_atomic = Arc::new(AtomicU32::new(u32::try_from(object_count).unwrap()));
        let force_acc = GpuBuffer::new(
            object_count,
            "force accumulator buffer",
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            &device,
        );

        spawn_simulation_thread(
            buffers.flags,
            buffers.masses,
            buffers.velocities,
            buffers.aabbs,
            buffers.bvh_nodes,
            force_acc,
            device.clone(),
            queue.clone(),
            self.event_loop_proxy.clone(),
            exit_notification_receiver.clone(),
            node_count_atomic.clone(),
        );

        self.gpu_state = Some(GpuState {
            shape_renderer,
            aabb_renderer,
            exit_notification_sender,
            world_aabb,
            object_count,
            camera_buffer,
            node_count_atomic,

            surface_config,
            queue,
            device,
            surface,
            window,
        });
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::RedrawRequested => {
                if let Some(state) = &self.gpu_state {
                    state.window.request_redraw();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.gpu_state {
                    state.surface_config.width = size.width;
                    state.surface_config.height = size.height;
                    state.surface.configure(&state.device, &state.surface_config);
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.gpu_state {
                    let view_size = state.window.inner_size();
                    let world_height = state.world_aabb.max().y - state.world_aabb.min().y;
                    let camera = orthographic_camera(self.render_parameters.zoom, view_size.cast(), world_height);
                    state.camera_buffer.write(&state.queue, &[Camera::new(camera)]);

                    let surface_texture =
                        state.surface.get_current_texture().expect("Failed to acquire next swap chain texture");
                    let surface_texture_view = surface_texture.texture.create_view(&TextureViewDescriptor::default());
                    let node_count = usize::try_from(state.node_count_atomic.load(Ordering::Relaxed)).unwrap();
                    render_scene(
                        surface_texture_view,
                        &self.render_parameters,
                        &mut state.shape_renderer,
                        &mut state.aabb_renderer,
                        0..state.object_count,
                        node_count,
                        &state.device,
                        &state.queue,
                    );

                    state.window.pre_present_notify();
                    surface_texture.present();
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
            } => self.render_parameters.enabled = !self.render_parameters.enabled,

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyA),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.render_parameters.draw_aabbs = !self.render_parameters.draw_aabbs;
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

            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, dy),
                ..
            } => {
                self.render_parameters.zoom *= 1.0 + dy * 0.2;
            }

            _ => (),
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.gpu_state {
            let _ = state.exit_notification_sender.try_send(());
        }
    }
}

fn init_wgpu(
    wgpu: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue, TextureFormat) {
    let adapter = block_on(wgpu.request_adapter(&RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::None),
        force_fallback_adapter: false,
        compatible_surface: Some(surface),
    }))
    .expect("Failed to find an appropriate adapter");

    let mut required_limits = wgpu::Limits::defaults().using_resolution(adapter.limits());
    required_limits.max_push_constant_size = u32::try_from(size_of::<CombineNodePass>()).unwrap();
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::PIPELINE_CACHE
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::PUSH_CONSTANTS
            | wgpu::Features::POLYGON_MODE_LINE,
        required_limits,
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
    surface_texture_view: TextureView,
    render_parameters: &RenderParameters,
    shape_renderer: &mut ShapeRenderer,
    aabb_renderer: &mut AabbRenderer,
    range: Range<usize>,
    node_count: usize,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> SubmissionIndex {
    let pass_duration_measurer = PassDurationMeasurer::new(device);
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    let measurement_start = pass_duration_measurer.start(&mut encoder);

    let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &surface_texture_view,
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
    if render_parameters.enabled {
        shape_renderer.render(&mut render_pass, range.clone());
    }
    if render_parameters.draw_aabbs {
        aabb_renderer.render(&mut render_pass, 0..node_count);
    }
    // TODO: EDF
    drop(render_pass);

    let measurement_result = measurement_start.finish(&mut encoder);
    let submission_index = queue.submit([encoder.finish()]);

    let duration = measurement_result.duration();
    println!("Rendered {} objects in {:?}", range.len(), duration);

    submission_index
}

fn orthographic_camera(zoom: f32, view_size: PhysicalSize<f32>, world_height: f32) -> [[f32; 4]; 4] {
    let aspect = view_size.width / view_size.height;
    let world_width = world_height * aspect;
    let l = -world_width * 0.5;
    let r = world_width * 0.5;
    let b = -world_height * 0.5;
    let t = world_height * 0.5;
    // TODO: implement zoom
    let sx = zoom * 2.0 / (r - l);
    let sy = zoom * 2.0 / (t - b);
    [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn spawn_simulation_thread(
    flags: GpuBuffer<Flags>,
    masses: GpuBuffer<Mass>,
    velocities: GpuBuffer<Velocity>,
    aabbs: GpuBuffer<AABB>,
    bvh_nodes: GpuBuffer<BvhNode>,
    foce_acc: GpuBuffer<[f32; 2]>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    event_loop_proxy: EventLoopProxy<AppEvent>,
    exit_notification_receiver: crossbeam::channel::Receiver<()>,
    node_count_atomic: Arc<AtomicU32>,
) {
    thread::spawn(move || {
        let mut last_redraw = Instant::now();
        let dt = GpuBuffer::new(1, "dt buffer", BufferUsages::UNIFORM | BufferUsages::COPY_DST, &device);
        dt.write(&queue, &[0.0001]);

        let object_count = flags.len();
        let mut bvh_builder = BvhBuilder::new(&device, aabbs.clone(), bvh_nodes.clone(), object_count);
        let integrator = GpuIntegrator::new(&device, dt, flags, masses, velocities, aabbs, bvh_nodes, foce_acc);
        let integration_duration_measurer = PassDurationMeasurer::new(&device);
        let bvh_duration_measurer = PassDurationMeasurer::new(&device);

        loop {
            if exit_notification_receiver.try_recv().is_ok() {
                break;
            }

            let now = Instant::now();
            if now - last_redraw >= Duration::from_secs_f32(1.0 / 60.0) {
                last_redraw = now;
                event_loop_proxy.send_event(AppEvent::RedrawRequested).unwrap();
            }

            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());

            let integration_measurement_start = integration_duration_measurer.start(&mut encoder);
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            integrator.compute(&mut compute_pass);
            drop(compute_pass);
            let integration_measurement_result = integration_measurement_start.finish(&mut encoder);

            let bvh_measurement_start = bvh_duration_measurer.start(&mut encoder);
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            bvh_builder.compute(&mut compute_pass);
            let node_count = bvh_builder.node_count();
            drop(compute_pass);
            let bvh_measurement_result = bvh_measurement_start.finish(&mut encoder);

            let submission_index = queue.submit([encoder.finish()]);
            node_count_atomic.store(node_count, Ordering::SeqCst);
            device.wait_for_submission(submission_index).unwrap();

            let integration_duration = integration_measurement_result.duration();
            println!("Integrated {} objects in {:?}", object_count, integration_duration);
            let bvh_duration = bvh_measurement_result.duration();
            println!("Built BVH with {} nodes in {:?}", node_count, bvh_duration);
        }
    });
}
