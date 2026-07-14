#![allow(clippy::too_many_arguments)]

pub mod aabb_renderer;
pub mod assign_object_cells;
pub mod buffer_sets;
pub mod calculate_cell_offsets;
pub mod calculate_cell_offsets_dispatch_dimensions;
pub mod calculate_grid_aabb;
pub mod collision_broad_phase_grid;
pub mod collision_narrow_phase;
pub mod collision_narrow_phase_dispatch_dimensions;
pub mod command_timings;
pub mod compute_stage;
pub mod config;
pub mod device_buffer;
pub mod integrator;
pub mod objects;
pub mod phase_state;
pub mod populate_grid_cells;
pub mod reset_grid_aabb;
pub mod scene;
pub mod shaders;
pub mod shape_renderer;
pub mod util;

use core::panic;
use std::{
    io::{Write as _, stdout},
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossbeam::channel;
use image::{ImageBuffer, Rgba};
use nalgebra::Vector2;
use pollster::block_on;
use shaders::common::Mass;
use wgpu::{
    Adapter, BufferUsages, CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor, CurrentSurfaceTexture,
    Device, DeviceDescriptor, Extent3d, InstanceDescriptor, Origin3d, PollType, PowerPreference, PresentMode, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RequestAdapterOptions, Surface, SurfaceConfiguration,
    TexelCopyBufferInfo, TexelCopyTextureInfo, TextureAspect, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor, wgt::TextureDescriptor,
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
    assign_object_cells::AssignObjectCells,
    buffer_sets::BroadPhaseBuffers,
    calculate_cell_offsets::CalculateCellOffsets,
    calculate_cell_offsets_dispatch_dimensions::CellIterationDispatchDimensions,
    calculate_grid_aabb::CalculateGridAABB,
    collision_broad_phase_grid::CollisionBroadPhaseGrid,
    collision_narrow_phase::NarrowPhase,
    collision_narrow_phase_dispatch_dimensions::NarrowPhaseDispatchIndirectArgs,
    compute_stage::ComputeStage,
    config::CONFIG,
    device_buffer::DeviceBuffer,
    integrator::Integrator,
    objects::Objects,
    phase_state::{PhaseStateRing, PhaseStateRingConfig},
    populate_grid_cells::PopulateGridCells,
    reset_grid_aabb::ResetGridAABB,
    scene::create_scene,
    shaders::{
        calculate_cell_offsets_dispatch_dimensions::N_CELL_INDIRECT_DISPATCHES,
        common::{AABB, Camera, Color, MAX_CANDIDATES_PER_OBJECT, MAX_OBJECTS_PER_CELL, Shape},
    },
    shape_renderer::ShapeRenderer,
};

const UNIFORM: BufferUsages = BufferUsages::UNIFORM;
const STORAGE: BufferUsages = BufferUsages::STORAGE;
const COPY_DST: BufferUsages = BufferUsages::COPY_DST;
const COPY_SRC: BufferUsages = BufferUsages::COPY_SRC;
const INDIRECT: BufferUsages = BufferUsages::INDIRECT;
const MAP_READ: BufferUsages = BufferUsages::MAP_READ;

fn main() {
    println!("{:#?}", *CONFIG);

    let wgpu_instance = wgpu::Instance::new(InstanceDescriptor::new_without_display_handle());
    let (adapter, device, queue) = init_wgpu(&wgpu_instance);
    println!("{:#?}", adapter.get_info());

    let mut objects = Objects::default();
    // Dimensions have to be positive, refer to calculate_grid_aabb.wgsl
    let world_aabb = AABB {
        min: [0.0, 0.0],
        max: CONFIG.world_size(),
    };
    let world_size = world_aabb.size();
    println!("World size: {}x{}", world_size.x, world_size.y);
    create_scene(&mut objects, world_aabb);

    let object_count: u32 = objects.flags.len().try_into().unwrap();
    println!("Object count: {}", object_count);

    stdout().flush().unwrap();

    let object_count_buffer = DeviceBuffer::from_data(&device, &[object_count], "object_count", UNIFORM);
    let spectrum_width = DeviceBuffer::from_data(&device, &[CONFIG.spectrum_width], "energy spectrum width", UNIFORM);
    let masses = DeviceBuffer::from_data(&device, &objects.masses, "masses", STORAGE | COPY_DST);
    let colors = DeviceBuffer::from_data(&device, &objects.colors, "colors", STORAGE | COPY_DST);
    let shapes = DeviceBuffer::from_data(&device, &objects.shapes, "shapes", STORAGE | COPY_DST);
    let phase_state_ring_config = PhaseStateRingConfig {
        n_frames: CONFIG.n_frames,
        n_compute: CONFIG.n_compute,
    };
    let phase_state_ring = Arc::new(Mutex::new(PhaseStateRing::new(
        phase_state_ring_config,
        &device,
        object_count,
        &objects.flags,
        &objects.aabbs,
        &objects.velocities,
    )));

    let exit_requested = Arc::new(AtomicBool::new(false));
    let event_loop = if CONFIG.headless {
        None
    } else {
        Some(EventLoop::with_user_event().build().expect("Failed to create event loop"))
    };
    let event_loop_proxy = event_loop.as_ref().map(|event_loop| event_loop.create_proxy());

    let render_parameters = RenderParameters {
        offset: Vector2::new(world_aabb.size().x * 0.5, -world_aabb.size().y * 0.5),
        ..Default::default()
    };
    let pause_simulation = Arc::new(AtomicBool::new(false));
    let sim_join_handle = spawn_simulation_thread(
        device.clone(),
        queue.clone(),
        object_count,
        object_count_buffer,
        world_aabb,
        phase_state_ring_config,
        phase_state_ring.clone(),
        spectrum_width.clone(),
        colors.clone(),
        shapes.clone(),
        masses.clone(),
        exit_requested.clone(),
        event_loop_proxy.clone(),
        render_parameters,
        pause_simulation.clone(),
    );

    let poll_join_handle = thread::spawn({
        let device = device.clone();
        let exit_requested = exit_requested.clone();
        move || {
            loop {
                device.poll(PollType::Poll).unwrap();
                if exit_requested.load(Ordering::Relaxed) {
                    break;
                }
            }
        }
    });

    if let Some(event_loop) = event_loop {
        let mut app = App {
            wgpu_instance,
            adapter,
            device,
            queue,
            sim_state: None,
            render_parameters,
            world_aabb,
            cursor_position: None,
            object_count,
            spectrum_width: spectrum_width.clone(),
            colors,
            shapes,
            masses,
            phase_state_ring_config,
            phase_state_ring,
            exit_requested,
            desired_maximum_frame_latency: phase_state_ring_config.n_frames,
            pause_simulation: pause_simulation.clone(),
        };
        event_loop.run_app(&mut app).expect("Failed to run app");
    } else {
        sim_join_handle.join().unwrap();
        poll_join_handle.join().unwrap();
    }
}

struct App<'a> {
    wgpu_instance: wgpu::Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    sim_state: Option<SimState<'a>>,
    render_parameters: RenderParameters,
    world_aabb: AABB,
    cursor_position: Option<Vector2<f32>>,
    object_count: u32,
    spectrum_width: DeviceBuffer<f32>,
    colors: DeviceBuffer<Color>,
    shapes: DeviceBuffer<Shape>,
    masses: DeviceBuffer<Mass>,
    phase_state_ring_config: PhaseStateRingConfig,
    phase_state_ring: Arc<Mutex<PhaseStateRing>>,
    exit_requested: Arc<AtomicBool>,
    desired_maximum_frame_latency: usize,
    pause_simulation: Arc<AtomicBool>,
}

#[derive(Copy, Clone, Debug)]
enum AppEvent {
    RedrawRequested,
    ExitEventLoop,
}

#[derive(Copy, Clone)]
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
            zoom: 1.0,
            offset: Vector2::new(0.0, 0.0),
        }
    }
}

struct SimState<'a> {
    shape_renderer: ShapeRenderer,
    aabb_renderer: AabbRenderer,
    camera: DeviceBuffer<Camera>,
    window: Arc<Window>,
    surface: Surface<'a>,
    surface_config: SurfaceConfiguration,
}

impl ApplicationHandler<AppEvent> for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = create_window(event_loop);
        let window_size = window.inner_size();
        if CONFIG.printouts {
            println!("Window size: {}x{}", window_size.width, window_size.height);
        }

        let surface = self.wgpu_instance.create_surface(window.clone()).unwrap();
        let (surface_config, swapchain_format) =
            init_surface(self.desired_maximum_frame_latency, &surface, &self.adapter, &self.device, &window);

        let camera = DeviceBuffer::<Camera>::new(&self.device, 1, "camera", UNIFORM | COPY_DST);
        let shape_renderer = ShapeRenderer::new(
            &self.device,
            swapchain_format,
            camera.clone(),
            self.spectrum_width.clone(),
            self.colors.clone(),
            self.shapes.clone(),
            self.masses.clone(),
            self.phase_state_ring_config,
        );
        let aabb_renderer = AabbRenderer::new(
            &self.device,
            swapchain_format,
            camera.clone(),
            self.object_count,
            self.phase_state_ring_config,
        );

        self.sim_state = Some(SimState {
            shape_renderer,
            aabb_renderer,
            camera,
            window,
            surface,
            surface_config,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.sim_state {
                    state.surface_config.width = size.width;
                    state.surface_config.height = size.height;
                    state.surface.configure(&self.device, &state.surface_config);
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.sim_state {
                    let view_size = state.window.inner_size();
                    let world_height = self.world_aabb.max().y - self.world_aabb.min().y;
                    let camera_matrix = orthographic_camera(view_size.cast(), world_height, &self.render_parameters);
                    state.camera.write(&self.queue, &[Camera::new(camera_matrix)]);

                    let CurrentSurfaceTexture::Success(surface_texture) = state.surface.get_current_texture() else {
                        panic!("Failed to get current surface texture");
                    };
                    let surface_texture_view = surface_texture.texture.create_view(&TextureViewDescriptor::default());
                    render_scene(
                        surface_texture_view,
                        &self.render_parameters,
                        &mut state.shape_renderer,
                        &mut state.aabb_renderer,
                        &self.phase_state_ring,
                        0..self.object_count,
                        &self.device,
                        &self.queue,
                        |_| {},
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

            WindowEvent::KeyboardInput { event, .. } if key_pressed(&event, KeyCode::Space) => {
                self.pause_simulation.fetch_not(Ordering::SeqCst);
            }

            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, dy),
                ..
            } => {
                if let Some(state) = &self.sim_state {
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
        self.exit_requested.store(true, Ordering::SeqCst);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::RedrawRequested => {
                if let Some(state) = &self.sim_state {
                    state.window.request_redraw();
                }
            }

            AppEvent::ExitEventLoop => event_loop.exit(),
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

fn init_wgpu(instance: &wgpu::Instance) -> (Adapter, Device, Queue) {
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::from_env().unwrap_or(PowerPreference::HighPerformance),
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("Failed to find an appropriate adapter");

    let required_features = wgpu::Features::POLYGON_MODE_LINE
        | wgpu::Features::IMMEDIATES
        | wgpu::Features::SUBGROUP
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
    let required_limits = wgpu::Limits {
        max_immediate_size: 32,
        max_storage_buffers_per_shader_stage: 16,
        max_buffer_size: 1024 * 1024 * 1024,
        max_storage_buffer_binding_size: 1024 * 1024 * 1024,
        ..wgpu::Limits::defaults().using_resolution(adapter.limits())
    };
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
        label: None,
        required_features,
        required_limits,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .expect("Failed to create device");

    (adapter, device, queue)
}

fn init_surface(
    desired_maximum_frame_latency: usize,
    surface: &Surface,
    adapter: &Adapter,
    device: &Device,
    window: &Window,
) -> (SurfaceConfiguration, TextureFormat) {
    let window_size = window.inner_size();
    let surface_config = SurfaceConfiguration {
        present_mode: PresentMode::AutoVsync,
        desired_maximum_frame_latency: u32::try_from(desired_maximum_frame_latency).unwrap(),
        ..surface.get_default_config(adapter, window_size.width, window_size.height).unwrap()
    };
    surface.configure(device, &surface_config);
    let swapchain_capabilities = surface.get_capabilities(adapter);
    let swapchain_format = swapchain_capabilities.formats[0];
    (surface_config, swapchain_format)
}

fn render_scene(
    surface_texture_view: TextureView,
    render_parameters: &RenderParameters,
    shape_renderer: &mut ShapeRenderer,
    aabb_renderer: &mut AabbRenderer,
    phase_state_ring: &Arc<Mutex<PhaseStateRing>>,
    instances: Range<u32>,
    device: &Device,
    queue: &Queue,
    before_submit: impl FnOnce(&mut CommandEncoder),
) {
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
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
        multiview_mask: None,
    });

    let mut phase_state_ring_guard = phase_state_ring.lock().unwrap();
    let current_frame_index = phase_state_ring_guard.current_frame_index();
    let current_frame = phase_state_ring_guard.current_frame().clone();
    phase_state_ring_guard.advance_frame();
    drop(phase_state_ring_guard);

    if render_parameters.enabled {
        shape_renderer.prepare(current_frame_index, device, &current_frame);
        shape_renderer.render(&mut render_pass, instances.clone());
    }
    if render_parameters.draw_aabbs {
        aabb_renderer.prepare(current_frame_index, device, &current_frame);
        aabb_renderer.render(&mut render_pass);
    }

    drop(render_pass);

    before_submit(&mut encoder);
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
    device: Device,
    queue: Queue,
    object_count: u32,
    object_count_buffer: DeviceBuffer<u32>,
    world_aabb: AABB,
    phase_state_ring_config: PhaseStateRingConfig,
    phase_state_ring: Arc<Mutex<PhaseStateRing>>,
    spectrum_width: DeviceBuffer<f32>,
    colors: DeviceBuffer<Color>,
    shapes: DeviceBuffer<Shape>,
    masses: DeviceBuffer<Mass>,
    exit_requested: Arc<AtomicBool>,
    event_loop_proxy: Option<EventLoopProxy<AppEvent>>,
    render_parameters: RenderParameters,
    pause_simulation: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn({
        const EXPORT_FRAME_SIZE: PhysicalSize<u32> = PhysicalSize::new(1920, 1080);

        let dt: f32 = CONFIG.dt;
        let dt_buffer = DeviceBuffer::from_data(&device, &[dt], "dt", UNIFORM);
        let max_candidates = object_count * MAX_CANDIDATES_PER_OBJECT;
        let candidates = DeviceBuffer::new(&device, max_candidates, "candidates", STORAGE | COPY_SRC);
        let candidate_count = DeviceBuffer::new(&device, 1, "candidate_count", STORAGE | UNIFORM | COPY_DST | COPY_SRC);
        let grid_min_x = DeviceBuffer::new(&device, 1, "grid min x", STORAGE | UNIFORM | COPY_SRC);
        let grid_min_y = DeviceBuffer::new(&device, 1, "grid min y", STORAGE | UNIFORM | COPY_SRC);
        let grid_max_x = DeviceBuffer::new(&device, 1, "grid max x", STORAGE | UNIFORM | COPY_SRC);
        let grid_max_y = DeviceBuffer::new(&device, 1, "grid max y", STORAGE | UNIFORM | COPY_SRC);
        let cell_size = DeviceBuffer::from_data(
            &device,
            &[CONFIG.particle_radius * 2.0],
            "cell size",
            STORAGE | UNIFORM | COPY_SRC,
        );
        let grid_size_x = DeviceBuffer::new(&device, 1, "grid size x", STORAGE | UNIFORM | COPY_SRC);
        let grid_size_y = DeviceBuffer::new(&device, 1, "grid size y", STORAGE | UNIFORM | COPY_SRC);
        let first_aabb = DeviceBuffer::new(&device, 1, "first aabb", UNIFORM | COPY_DST);
        let object_cells = DeviceBuffer::new(&device, object_count, "object cells", STORAGE | COPY_SRC);
        let max_cells = object_count * 10; // TODO: calculate max_cells properly
        let cell_object_count =
            DeviceBuffer::new(&device, max_cells, "cell object count", STORAGE | UNIFORM | COPY_DST | COPY_SRC);
        let dispatch_dimensions = DeviceBuffer::new(
            &device,
            N_CELL_INDIRECT_DISPATCHES,
            "cell offsets dispatch dimensions",
            STORAGE | INDIRECT | COPY_DST,
        );
        let current_cell_offset = DeviceBuffer::new(&device, 1, "current cell offset", STORAGE | COPY_SRC | COPY_DST);
        let cell_offsets = DeviceBuffer::new(&device, max_cells, "cell offsets", STORAGE);
        let cells = DeviceBuffer::new(&device, object_count * MAX_OBJECTS_PER_CELL, "cells", STORAGE);
        let forces = DeviceBuffer::new(&device, object_count * 2, "collision forces", STORAGE | COPY_SRC | COPY_DST);
        let stiffness = DeviceBuffer::from_data(&device, &[CONFIG.stiffness], "stiffness", UNIFORM | COPY_SRC);
        let restitution = DeviceBuffer::from_data(&device, &[CONFIG.restitution], "restitution", UNIFORM | COPY_SRC);
        let safety_margin = CONFIG.particle_radius * 2.0;
        let constraints = AABB {
            min: [world_aabb.min[0] + safety_margin, world_aabb.min[1] + safety_margin],
            max: [world_aabb.max[0] - safety_margin, world_aabb.max[1] - safety_margin],
        };
        let constraints_buffer = DeviceBuffer::from_data(&device, &[constraints], "constraints", UNIFORM | COPY_SRC);
        let broad_phase_buffers = BroadPhaseBuffers {
            first_aabb: first_aabb.clone(),
            grid_min_x,
            grid_max_x,
            grid_min_y,
            grid_max_y,
            cell_size,
            grid_size_x,
            grid_size_y,
            object_cells,
            current_cell_offset: current_cell_offset.clone(),
            cell_object_count: cell_object_count.clone(),
            cell_offsets,
            cells,
            candidates,
            candidate_count: candidate_count.clone(),
            masses: masses.clone(),
            forces: forces.clone(),
        };

        let reset_grid_aabb = ResetGridAABB::new(&device, &broad_phase_buffers);
        let mut calculate_grid_aabb = CalculateGridAABB::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            &broad_phase_buffers,
            phase_state_ring_config,
        );
        let cell_offsets_dispatch_dimensions =
            CellIterationDispatchDimensions::new(&device, &broad_phase_buffers, dispatch_dimensions.clone());
        let mut assign_object_cells = AssignObjectCells::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            &broad_phase_buffers,
            phase_state_ring_config,
        );
        let calculate_cell_offsets =
            CalculateCellOffsets::new(&device, dispatch_dimensions.clone(), &broad_phase_buffers);
        let populate_grid_cells =
            PopulateGridCells::new(&device, object_count, object_count_buffer.clone(), &broad_phase_buffers);
        let mut broad_phase_grid = CollisionBroadPhaseGrid::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            &broad_phase_buffers,
            phase_state_ring_config,
        );
        let narrow_phase_dispatch_dimensions = NarrowPhaseDispatchIndirectArgs::new(
            &device,
            broad_phase_buffers.candidate_count.clone(),
            dispatch_dimensions.clone(),
        );
        let mut narrow_phase = NarrowPhase::new(
            &device,
            dispatch_dimensions.clone(),
            stiffness.clone(),
            restitution.clone(),
            &broad_phase_buffers,
            masses.clone(),
            forces.clone(),
            phase_state_ring_config,
        );
        let mut integrator = Integrator::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            constraints_buffer.clone(),
            dt_buffer,
            masses.clone(),
            forces.clone(),
            phase_state_ring_config,
        );
        let timestamp_period = queue.get_timestamp_period();
        let (tx, rx) = channel::bounded(phase_state_ring_config.n_compute);
        let mut sim_step_count: usize = 0;
        let mut compute_submitted: usize = 0;
        let mut last_frame_instant = Instant::now();
        let start_instant = Instant::now();

        let export_frame_texture = device.create_texture(&TextureDescriptor {
            label: Some("Frame export"),
            size: Extent3d {
                width: EXPORT_FRAME_SIZE.width,
                height: EXPORT_FRAME_SIZE.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let world_height = world_aabb.max().y - world_aabb.min().y;
        let camera_matrix = orthographic_camera(EXPORT_FRAME_SIZE.cast(), world_height, &render_parameters);
        let camera_buffer =
            DeviceBuffer::<Camera>::from_data(&device, &[Camera::new(camera_matrix)], "camera", UNIFORM | COPY_DST);
        let mut shape_renderer = ShapeRenderer::new(
            &device,
            export_frame_texture.format(),
            camera_buffer.clone(),
            spectrum_width.clone(),
            colors.clone(),
            shapes.clone(),
            masses.clone(),
            phase_state_ring_config,
        );
        let mut aabb_renderer = AabbRenderer::new(
            &device,
            export_frame_texture.format(),
            camera_buffer.clone(),
            object_count,
            phase_state_ring_config,
        );
        let padded_bytes_per_row = (EXPORT_FRAME_SIZE.width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let export_frame_staging_buffer = DeviceBuffer::<u8>::new(
            &device,
            padded_bytes_per_row * EXPORT_FRAME_SIZE.height,
            "export frame staging buffer",
            COPY_DST | MAP_READ,
        );
        if let Some(output_path) = &CONFIG.output_path {
            std::fs::create_dir_all(output_path).unwrap();
        }

        move || loop {
            if exit_requested.load(Ordering::Relaxed) {
                break;
            }

            let sim_time = sim_step_count as f32 * dt;
            let real_time = start_instant.elapsed().as_secs_f32();
            let print_sim_rate =
                move || println!("Simulation rate: {} (sim {sim_time} / real {real_time})", sim_time / real_time);

            if CONFIG.sim_time_limit.is_some_and(|max_sim_time| sim_time >= max_sim_time) {
                println!("Simulation time limit reached");
                print_sim_rate();
                stdout().flush().unwrap();
                exit_requested.store(true, Ordering::SeqCst);
                if let Some(event_loop_proxy) = &event_loop_proxy
                    && CONFIG.exit_at_limit
                {
                    event_loop_proxy.send_event(AppEvent::ExitEventLoop).unwrap();
                }
                break;
            }

            if let Some(event_loop_proxy) = &event_loop_proxy
                && last_frame_instant.elapsed() >= Duration::from_secs_f32(1.0 / CONFIG.fps)
            {
                event_loop_proxy.send_event(AppEvent::RedrawRequested).unwrap();
                last_frame_instant = Instant::now();
            }

            if pause_simulation.load(Ordering::Relaxed) {
                thread::yield_now();
                continue;
            }

            // Set integrator buffers, advance the 2-state sliding window that the integrator uses
            let mut phase_state_ring_guard = phase_state_ring.lock().unwrap();
            let phase_state_index = phase_state_ring_guard.current_compute_index();
            let mut current_phase_state = phase_state_ring_guard.current_compute().clone();
            let next_phase_state = phase_state_ring_guard.next_compute().clone();
            phase_state_ring_guard.advance_compute();
            drop(phase_state_ring_guard);

            // Prepare the compute pass
            calculate_grid_aabb.prepare(&device, phase_state_index, &current_phase_state);
            assign_object_cells.prepare(&device, phase_state_index, &current_phase_state);
            broad_phase_grid.prepare(&device, phase_state_index, &current_phase_state);
            narrow_phase.prepare(&device, phase_state_index, &current_phase_state);
            integrator.prepare(&device, phase_state_index, &current_phase_state, &next_phase_state);

            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());

            // Reset buffers
            first_aabb.copy(0..1, current_phase_state.aabbs(), 0..1, &mut encoder);
            let timings = current_phase_state.command_timings();
            timings.measure(&mut encoder, "Clear buffers", |encoder| {
                encoder.clear_buffer(cell_object_count.buffer(), 0, None);
                encoder.clear_buffer(dispatch_dimensions.buffer(), 0, None);
                encoder.clear_buffer(current_cell_offset.buffer(), 0, None);
                encoder.clear_buffer(candidate_count.buffer(), 0, None);
                encoder.clear_buffer(forces.buffer(), 0, None);
            });

            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

            // Broad phase
            reset_grid_aabb.compute(&mut compute_pass, timings);
            calculate_grid_aabb.compute(&mut compute_pass, timings);
            cell_offsets_dispatch_dimensions.compute(&mut compute_pass, timings);
            assign_object_cells.compute(&mut compute_pass, timings);
            calculate_cell_offsets.compute(&mut compute_pass, timings);
            populate_grid_cells.compute(&mut compute_pass, timings);
            broad_phase_grid.compute(&mut compute_pass, timings);

            // Narrow phase
            narrow_phase_dispatch_dimensions.compute(&mut compute_pass, timings);
            narrow_phase.compute(&mut compute_pass, timings);

            // Integrate
            integrator.compute(&mut compute_pass, timings);

            drop(compute_pass);

            let timings_reader = timings.resolve(&mut encoder, timestamp_period);

            // Submit work
            let start = Instant::now();
            queue.submit([encoder.finish()]);
            queue.on_submitted_work_done({
                let tx = tx.clone();
                move || {
                    if CONFIG.printouts {
                        timings_reader.read(|timings| {
                            for (label, duration) in timings {
                                println!("{}: {:?}", label, duration);
                            }
                        });
                        println!("compute done in {:?}", start.elapsed());
                        print_sim_rate();
                    }
                    let _ = tx.send(());
                }
            });

            // Max PhaseStateRing::N_COMPUTE - 1 integrations at a time
            compute_submitted += 1;
            if compute_submitted >= phase_state_ring_config.n_compute - 1 {
                rx.recv().unwrap();
                compute_submitted -= 1;
            }

            // Print stats
            sim_step_count += 1;

            if let Some(output_path) = &CONFIG.output_path {
                // Render to the export texture
                let texture_view = export_frame_texture.create_view(&Default::default());
                render_scene(
                    texture_view,
                    &render_parameters,
                    &mut shape_renderer,
                    &mut aabb_renderer,
                    &phase_state_ring,
                    0..object_count,
                    &device,
                    &queue,
                    {
                        let export_frame_texture = export_frame_texture.clone();
                        let export_frame_staging_buffer = export_frame_staging_buffer.clone();
                        move |encoder| {
                            encoder.copy_texture_to_buffer(
                                TexelCopyTextureInfo {
                                    texture: &export_frame_texture,
                                    mip_level: 0,
                                    origin: Origin3d::ZERO,
                                    aspect: TextureAspect::All,
                                },
                                TexelCopyBufferInfo {
                                    buffer: export_frame_staging_buffer.buffer(),
                                    layout: wgpu::TexelCopyBufferLayout {
                                        offset: 0,
                                        bytes_per_row: Some(padded_bytes_per_row),
                                        rows_per_image: Some(EXPORT_FRAME_SIZE.height),
                                    },
                                },
                                Extent3d {
                                    width: EXPORT_FRAME_SIZE.width,
                                    height: EXPORT_FRAME_SIZE.height,
                                    depth_or_array_layers: 1,
                                },
                            );
                        }
                    },
                );
                let (tx, rx) = channel::bounded(1);
                queue.on_submitted_work_done({
                    let export_frame_staging_buffer = export_frame_staging_buffer.clone();
                    move || {
                        assert!(padded_bytes_per_row == EXPORT_FRAME_SIZE.width * 4);
                        let frame_size_in_bytes = padded_bytes_per_row * EXPORT_FRAME_SIZE.height;
                        export_frame_staging_buffer.read(frame_size_in_bytes.try_into().unwrap(), move |result| {
                            let data = result.unwrap();
                            let _ = tx.send(data);
                        });
                    }
                });
                let data = rx.recv().unwrap();
                rayon::spawn(move || {
                    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(
                        EXPORT_FRAME_SIZE.width,
                        EXPORT_FRAME_SIZE.height,
                        data.as_slice(),
                    )
                    .expect("invalid image size");
                    image.save(format!("{output_path}/{:06}.png", sim_step_count)).unwrap();
                });
            }
        }
    })
}
