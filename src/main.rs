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
pub mod phase_state;
pub mod scene;
pub mod shaders;
pub mod shape_renderer;
#[allow(unused)]
pub mod util;

use crate::{
    aabb_renderer::AabbRenderer,
    bvh_builder::{BvhBuildParameters, BvhBuilder},
    gpu_buffer::GpuBuffer,
    integration::GpuIntegrator,
    objects::Objects,
    phase_state::PhaseStateRing,
    scene::create_scene,
    shaders::{
        bvh::CombineNodePass,
        common::{AABB, BvhNode, Camera},
    },
    shape_renderer::ShapeRenderer,
};
use crossbeam::channel;
use pollster::block_on;
use shaders::common::Mass;
use std::{
    mem::size_of,
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self},
    time::Duration,
};
use wgpu::{
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, PollType, PresentMode, RenderPassColorAttachment,
    RenderPassDescriptor, RequestAdapterOptions, TextureFormat, TextureView, TextureViewDescriptor,
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
    _event_loop_proxy: EventLoopProxy<AppEvent>,
}

impl App<'_> {
    fn new(event_loop_proxy: EventLoopProxy<AppEvent>) -> Self {
        Self {
            render_parameters: RenderParameters::default(),
            gpu_state: None,
            _event_loop_proxy: event_loop_proxy,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum AppEvent {}

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
            zoom: 0.8,
        }
    }
}

struct GpuState<'a> {
    shape_renderer: ShapeRenderer,
    aabb_renderer: AabbRenderer,
    exit_requested: Arc<AtomicBool>,
    world_aabb: AABB,
    object_count: usize,
    camera: GpuBuffer<Camera>,
    phase_state_ring: Arc<Mutex<PhaseStateRing>>,

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
        // TODO: refactor, too much noise
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
        let object_count = objects.flags.len();
        let bvh_build_params = BvhBuildParameters::new(object_count);
        // TODO: don't store leaves
        let storage_copy_dst: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let nodes = GpuBuffer::new(bvh_build_params.node_count(), "bvh node buffer", storage_copy_dst, &device);
        nodes.write_iter(&queue, (0..u32::try_from(object_count).unwrap()).map(BvhNode::new));
        let bvh_builder = BvhBuilder::new(bvh_build_params, &device, nodes.clone());
        let node_count = bvh_builder.node_count();

        // TODO: split AABBs of objects and nodes
        let aabbs = GpuBuffer::new(node_count, "aabb buffer", storage_copy_dst, &device);
        aabbs.write(&queue, &objects.aabbs);

        let masses = GpuBuffer::from_data(&objects.masses, "mass buffer", storage_copy_dst, &device);
        let colors = GpuBuffer::from_data(&objects.colors, "color buffer", storage_copy_dst, &device);
        let shapes = GpuBuffer::from_data(&objects.shapes, "shape buffer", storage_copy_dst, &device);

        let phase_state_ring =
            Arc::new(Mutex::new(PhaseStateRing::new(&device, &objects.flags, &objects.aabbs, &objects.velocities)));

        println!("Window size: {}x{}", window_size.width, window_size.height);
        println!("Object count: {}", object_count);

        let camera =
            GpuBuffer::<Camera>::new(1, "camera buffer", BufferUsages::UNIFORM | BufferUsages::COPY_DST, &device);
        let size_factor =
            GpuBuffer::new(1, "size factor buffer", BufferUsages::UNIFORM | BufferUsages::COPY_DST, &device);
        size_factor.write(&queue, &[1.0]);
        let shape_renderer =
            ShapeRenderer::new(&device, swapchain_format, camera.clone(), size_factor.clone(), colors, shapes);
        let node_count = u32::try_from(node_count).unwrap();
        let node_count_buffer =
            GpuBuffer::from_data(&[node_count], "node count buffer", BufferUsages::UNIFORM, &device);
        let aabb_renderer = AabbRenderer::new(&device, swapchain_format, camera.clone(), node_count);
        let exit_requested = Arc::new(AtomicBool::new(false));

        spawn_simulation_thread(
            object_count,
            phase_state_ring.clone(),
            masses,
            nodes,
            node_count_buffer,
            device.clone(),
            queue.clone(),
            exit_requested.clone(),
            bvh_builder,
        );

        thread::spawn({
            let device = device.clone();
            let exit_requested = exit_requested.clone();
            move || {
                loop {
                    device.poll(PollType::Poll).unwrap();
                    if exit_requested.load(Ordering::Relaxed) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(1));
                }
            }
        });

        self.gpu_state = Some(GpuState {
            shape_renderer,
            aabb_renderer,
            exit_requested,
            world_aabb,
            object_count,
            camera,
            phase_state_ring,

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
                    state.camera.write(&state.queue, &[Camera::new(camera)]);

                    let surface_texture =
                        state.surface.get_current_texture().expect("Failed to acquire next swap chain texture");
                    let surface_texture_view = surface_texture.texture.create_view(&TextureViewDescriptor::default());
                    render_scene(
                        surface_texture_view,
                        &self.render_parameters,
                        &mut state.shape_renderer,
                        &mut state.aabb_renderer,
                        &state.phase_state_ring,
                        0..state.object_count,
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
                self.render_parameters.zoom *= 1.0 + dy * 0.1;
            }

            _ => (),
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.gpu_state {
            state.exit_requested.store(true, Ordering::SeqCst);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.gpu_state {
            state.window.request_redraw();
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
    phase_state_ring: &Arc<Mutex<PhaseStateRing>>,
    range: Range<usize>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
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

    let phase_state_ring_guard = phase_state_ring.lock().unwrap();
    let phase_state_index = phase_state_ring_guard.current_index();
    let phase_state = phase_state_ring_guard.current().clone();
    drop(phase_state_ring_guard);

    if render_parameters.enabled {
        shape_renderer.prepare(phase_state_index, device, &phase_state);
        shape_renderer.render(&mut render_pass, range.clone());
    }
    if render_parameters.draw_aabbs {
        aabb_renderer.prepare(phase_state_index, device, &phase_state);
        aabb_renderer.render(&mut render_pass);
    }

    drop(render_pass);
    queue.submit([encoder.finish()]);
}

fn orthographic_camera(zoom: f32, view_size: PhysicalSize<f32>, world_height: f32) -> [[f32; 4]; 4] {
    let aspect = view_size.width / view_size.height;
    let world_width = world_height * aspect;
    let l = -world_width * 0.5;
    let r = world_width * 0.5;
    let b = -world_height * 0.5;
    let t = world_height * 0.5;
    // TODO: implement panning
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
    object_count: usize,
    phase_state_ring: Arc<Mutex<PhaseStateRing>>,
    masses: GpuBuffer<Mass>,
    nodes: GpuBuffer<BvhNode>,
    node_count_buffer: GpuBuffer<u32>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    exit_requested: Arc<AtomicBool>,
    mut bvh_builder: BvhBuilder,
) {
    thread::spawn({
        // let device = device.clone();
        let dt = GpuBuffer::from_data(&[0.001], "dt buffer", BufferUsages::UNIFORM, &device);
        let mut integrator = GpuIntegrator::new(&device, dt, masses, nodes, node_count_buffer.clone(), object_count);

        let (tx, rx) = channel::bounded(PhaseStateRing::CAPACITY);
        let mut frames_submitted = 0usize;

        move || loop {
            if exit_requested.load(Ordering::Relaxed) {
                break;
            }

            let mut phase_state_ring_guard = phase_state_ring.lock().unwrap();
            let phase_state_index = phase_state_ring_guard.current_index();
            let current_phase_state = phase_state_ring_guard.current().clone();
            let next_phase_state = phase_state_ring_guard.next().clone();
            phase_state_ring_guard.advance();
            drop(phase_state_ring_guard);

            bvh_builder.prepare(phase_state_index, &device, &current_phase_state);
            integrator.prepare(phase_state_index, &device, &current_phase_state, &next_phase_state);

            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("bvh pass"),
                timestamp_writes: None,
            });
            // TODO: batch updates?
            bvh_builder.compute(&mut compute_pass);
            integrator.compute(&mut compute_pass);
            drop(compute_pass);
            let command_buffer = encoder.finish();
            queue.submit([command_buffer]);
            queue.on_submitted_work_done({
                let tx = tx.clone();
                move || tx.send(()).unwrap()
            });
            device.poll(PollType::Poll).unwrap();

            frames_submitted += 1;
            if frames_submitted >= PhaseStateRing::CAPACITY {
                rx.recv().unwrap();
                frames_submitted -= 1;
            }
        }
    });
}
