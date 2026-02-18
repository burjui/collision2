#![allow(clippy::too_many_arguments)]

pub mod aabb_renderer;
pub mod bvh_builder;
pub mod gpu_buffer;
pub mod integration;
#[cfg(test)]
mod mock_bvh_test;
pub mod objects;
pub mod phase_state;
pub mod scene;
pub mod shaders;
pub mod shape_renderer;
pub mod util;

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

use crossbeam::channel;
use nalgebra::Vector2;
use pollster::block_on;
use shaders::common::Mass;
use wgpu::{
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, Device, DeviceDescriptor, InstanceDescriptor,
    PollType, PowerPreference, PresentMode, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RequestAdapterOptions, Surface, SurfaceConfiguration, TextureFormat, TextureView, TextureViewDescriptor,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    keyboard::KeyCode,
    window::{Fullscreen, Window, WindowAttributes, WindowId},
};

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

fn main() {
    let event_loop = EventLoop::with_user_event().build().expect("Failed to create event loop");
    let event_loop_proxy = event_loop.create_proxy();
    let mut app = App::new(event_loop_proxy);
    event_loop.run_app(&mut app).expect("Failed to run app");
}

struct App<'a> {
    render_parameters: RenderParameters,
    gpu_state: Option<GpuState<'a>>,
    world_aabb: AABB,
    _event_loop_proxy: EventLoopProxy<AppEvent>,
    cursor_position: Option<Vector2<f32>>,
}

impl App<'_> {
    fn new(_event_loop_proxy: EventLoopProxy<AppEvent>) -> Self {
        Self {
            render_parameters: RenderParameters::default(),
            world_aabb: AABB {
                min: [-1000.0, -1000.0],
                max: [1000.0, 1000.0],
            },
            gpu_state: None,
            _event_loop_proxy,
            cursor_position: None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum AppEvent {}

struct RenderParameters {
    enabled: bool,
    draw_aabbs: bool,
    zoom: f32,
    offset: Vector2<f32>,
}

impl Default for RenderParameters {
    fn default() -> Self {
        Self {
            enabled: true,
            draw_aabbs: false,
            zoom: 0.8,
            offset: Vector2::new(0.0, 0.0),
        }
    }
}

struct GpuState<'a> {
    shape_renderer: ShapeRenderer,
    aabb_renderer: AabbRenderer,
    exit_requested: Arc<AtomicBool>,
    object_count: usize,
    camera: GpuBuffer<Camera>,
    phase_state_ring: Arc<Mutex<PhaseStateRing>>,

    window: Arc<Window>,
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,
    device: Device,
    queue: Queue,
}

impl ApplicationHandler<AppEvent> for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = create_window(event_loop);
        let wgpu_instance = wgpu::Instance::new(&InstanceDescriptor::from_env_or_default());
        let surface = wgpu_instance.create_surface(window.clone()).unwrap();
        let (surface_config, device, queue, swapchain_format) = init_wgpu(&wgpu_instance, &window, &surface);

        let mut objects = Objects::default();
        create_scene(&mut objects, self.world_aabb);

        let object_count = objects.flags.len();
        println!("Object count: {}", object_count);
        let window_size = window.inner_size();
        println!("Window size: {}x{}", window_size.width, window_size.height);
        let world_size = self.world_aabb.size();
        println!("World size: {}x{}", world_size.x, world_size.y);

        let bvh_build_params = BvhBuildParameters::new(object_count);
        // TODO: don't store leaves
        let storage_copy_dst: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let node_count = bvh_build_params.node_count();
        let nodes = GpuBuffer::new(node_count, "bvh node buffer", storage_copy_dst, &device);
        nodes.write_iter(&queue, (0..u32::try_from(object_count).unwrap()).map(BvhNode::new));
        let bvh_builder = BvhBuilder::new(bvh_build_params, &device, nodes.clone());

        let masses = GpuBuffer::from_data(&objects.masses, "mass buffer", storage_copy_dst, &device);
        let colors = GpuBuffer::from_data(&objects.colors, "color buffer", storage_copy_dst, &device);
        let shapes = GpuBuffer::from_data(&objects.shapes, "shape buffer", storage_copy_dst, &device);

        let phase_state_ring = Arc::new(Mutex::new(PhaseStateRing::new(
            &device,
            &queue,
            &objects.flags,
            &objects.aabbs,
            &objects.velocities,
            node_count,
        )));

        let camera =
            GpuBuffer::<Camera>::new(1, "camera buffer", BufferUsages::UNIFORM | BufferUsages::COPY_DST, &device);
        let shape_renderer = ShapeRenderer::new(&device, swapchain_format, camera.clone(), colors, shapes);
        let aabb_renderer = AabbRenderer::new(&device, swapchain_format, camera.clone(), node_count);
        let exit_requested = Arc::new(AtomicBool::new(false));

        spawn_simulation_thread(
            object_count,
            device.clone(),
            queue.clone(),
            phase_state_ring.clone(),
            masses,
            nodes,
            bvh_builder,
            exit_requested.clone(),
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
            object_count,
            camera,
            phase_state_ring,

            window,
            surface,
            surface_config,
            device,
            queue,
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
                    let world_height = self.world_aabb.max().y - self.world_aabb.min().y;
                    let camera = orthographic_camera(view_size.cast(), world_height, &self.render_parameters);
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

            WindowEvent::KeyboardInput { event, .. } if key_pressed(&event, KeyCode::KeyR) => {
                self.render_parameters.enabled = !self.render_parameters.enabled
            }

            WindowEvent::KeyboardInput { event, .. } if key_pressed(&event, KeyCode::KeyA) => {
                self.render_parameters.draw_aabbs = !self.render_parameters.draw_aabbs
            }

            WindowEvent::KeyboardInput { event, .. } if key_pressed(&event, KeyCode::Escape) => event_loop.exit(),
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, dy),
                ..
            } => {
                if let Some(state) = &self.gpu_state {
                    if let Some(cursor_pos) = self.cursor_position {
                        let zoom_old = self.render_parameters.zoom;
                        let zoom_new = self.render_parameters.zoom * (1.0 + dy * 0.1);
                        let view_size = state.window.inner_size();
                        let view_center = Vector2::new(view_size.width as f32 / 2.0, view_size.height as f32 / 2.0);
                        let cursor_relative_to_center = cursor_pos - view_center;
                        let world_height = self.world_aabb.max().y - self.world_aabb.min().y;
                        let aspect = view_size.width as f32 / view_size.height as f32;
                        let world_width = world_height * aspect;
                        let view_width = view_size.width as f32;
                        let view_height = view_size.height as f32;
                        let cursor_offset_world_x = cursor_relative_to_center.x * world_width / view_width;
                        let cursor_offset_world_y = cursor_relative_to_center.y * world_height / view_height;
                        let zoom_ratio = zoom_new / zoom_old - 1.0;
                        self.render_parameters.zoom = zoom_new;
                        self.render_parameters.offset.x +=
                            (self.render_parameters.offset.x + cursor_offset_world_x) * zoom_ratio;
                        self.render_parameters.offset.y +=
                            (self.render_parameters.offset.y + cursor_offset_world_y) * zoom_ratio;
                    } else {
                        self.render_parameters.zoom *= 1.0 + dy * 0.1;
                    }
                } else {
                    self.render_parameters.zoom *= 1.0 + dy * 0.1;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let position: [f64; 2] = position.into();
                self.cursor_position = Some(Vector2::from(position).cast());
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

fn key_pressed(event: &KeyEvent, key: KeyCode) -> bool {
    event.state == ElementState::Pressed && event.physical_key == key
}

fn create_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    let mut window_attributes = WindowAttributes::default();
    window_attributes.inner_size = Some(PhysicalSize::new(1600, 800).into());
    window_attributes.fullscreen = Some(Fullscreen::Borderless(None));
    let window = event_loop.create_window(window_attributes).expect("Failed to create window");
    Arc::new(window)
}

fn init_wgpu(
    instance: &wgpu::Instance,
    window: &Window,
    surface: &Surface,
) -> (SurfaceConfiguration, Device, Queue, TextureFormat) {
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::from_env().unwrap_or(PowerPreference::None),
        force_fallback_adapter: false,
        compatible_surface: Some(surface),
    }))
    .expect("Failed to find an appropriate adapter");

    let required_features = wgpu::Features::PUSH_CONSTANTS | wgpu::Features::POLYGON_MODE_LINE;
    let required_limits = wgpu::Limits {
        max_push_constant_size: u32::try_from(size_of::<CombineNodePass>()).unwrap(),
        ..wgpu::Limits::defaults().using_resolution(adapter.limits())
    };
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
        label: None,
        required_features,
        required_limits,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    }))
    .expect("Failed to create device");

    let window_size = window.inner_size();
    let surface_config = wgpu::SurfaceConfiguration {
        present_mode: PresentMode::AutoVsync,
        desired_maximum_frame_latency: 4,
        ..surface.get_default_config(&adapter, window_size.width, window_size.height).unwrap()
    };
    surface.configure(&device, &surface_config);
    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];
    (surface_config, device, queue, swapchain_format)
}

fn render_scene(
    surface_texture_view: TextureView,
    render_parameters: &RenderParameters,
    shape_renderer: &mut ShapeRenderer,
    aabb_renderer: &mut AabbRenderer,
    phase_state_ring: &Arc<Mutex<PhaseStateRing>>,
    range: Range<usize>,
    device: &Device,
    queue: &Queue,
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

fn orthographic_camera(view_size: PhysicalSize<f32>, world_height: f32, params: &RenderParameters) -> [[f32; 4]; 4] {
    let aspect = view_size.width / view_size.height;
    let world_width = world_height * aspect;
    let left = -world_width * 0.5;
    let right = world_width * 0.5;
    let bottom = -world_height * 0.5;
    let top = world_height * 0.5;
    let sx = params.zoom * 2.0 / (right - left);
    let sy = params.zoom * 2.0 / (top - bottom);
    let tx = -params.offset.x * 2.0 / (right - left);
    let ty = params.offset.y * 2.0 / (top - bottom);
    [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [tx, ty, 0.0, 1.0],
    ]
}

fn spawn_simulation_thread(
    object_count: usize,
    device: Device,
    queue: Queue,
    phase_state_ring: Arc<Mutex<PhaseStateRing>>,
    masses: GpuBuffer<Mass>,
    nodes: GpuBuffer<BvhNode>,
    mut bvh_builder: BvhBuilder,
    exit_requested: Arc<AtomicBool>,
) {
    thread::spawn({
        // let device = device.clone();
        let dt = GpuBuffer::from_data(&[0.001], "dt buffer", BufferUsages::UNIFORM, &device);
        let mut integrator = GpuIntegrator::new(&device, dt, masses, nodes, object_count);

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
                move || {
                    let _ = tx.send(());
                }
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
