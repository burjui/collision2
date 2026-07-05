#![allow(clippy::too_many_arguments)]

pub mod aabb_renderer;
pub mod assign_object_cells;
pub mod calculate_cell_offsets;
pub mod calculate_cell_offsets_dispatch_dimensions;
pub mod calculate_grid_aabb;
pub mod collision_broad_phase_grid;
pub mod collision_forces_reset;
pub mod collision_narrow_phase;
pub mod collision_narrow_phase_dispatch_dimensions;
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
    io::Write as _,
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self},
    time::{Duration, Instant},
};

use crossbeam::channel;
use nalgebra::Vector2;
use pollster::block_on;
use shaders::common::Mass;
use wgpu::{
    Adapter, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, CurrentSurfaceTexture, Device,
    DeviceDescriptor, InstanceDescriptor, PollType, PowerPreference, PresentMode, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, RequestAdapterOptions, Surface, SurfaceConfiguration, TextureFormat, TextureView,
    TextureViewDescriptor,
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
    calculate_cell_offsets::CalculateCellOffsets,
    calculate_cell_offsets_dispatch_dimensions::CalculateCellIterationDispatchDimensions,
    calculate_grid_aabb::CalculateGridAABB,
    collision_broad_phase_grid::CollisionBroadPhaseGrid,
    collision_forces_reset::CollisionReset,
    collision_narrow_phase::NarrowPhase,
    collision_narrow_phase_dispatch_dimensions::NarrowPhaseDispatchIndirectArgsCalculator,
    config::CONFIG,
    device_buffer::DeviceBuffer,
    integrator::Integrator,
    objects::Objects,
    phase_state::{PhaseStateRing, PhaseStateRingConfig},
    populate_grid_cells::PopulateGridCells,
    reset_grid_aabb::ResetGridAABB,
    scene::{PARTICLE_RADIUS, create_scene},
    shaders::{
        calculate_cell_offsets_dispatch_dimensions::N_CELL_INDIRECT_DISPATCHES,
        common::{
            AABB, Camera, CellPosition, Color, DispatchIndirectArgs, MAX_CANDIDATES_PER_OBJECT, MAX_OBJECTS_PER_CELL,
            Shape,
        },
    },
    shape_renderer::ShapeRenderer,
};

fn main() {
    let wgpu_instance = wgpu::Instance::new(InstanceDescriptor::new_without_display_handle());
    let (adapter, device, queue) = init_wgpu(&wgpu_instance);

    let mut objects = Objects::default();
    let world_aabb = AABB {
        min: [-3200.0, -2000.0],
        max: [3200.0, 2000.0],
    };
    create_scene(&mut objects, world_aabb);

    let object_count: u32 = objects.flags.len().try_into().unwrap();
    if CONFIG.printouts {
        println!("Object count: {}", object_count);
    }

    let object_count_buffer: DeviceBuffer<u32> =
        DeviceBuffer::from_data(&device, &[object_count], "object_count", BufferUsages::UNIFORM);
    // TODO: don't store leaves
    let storage_copy_dst: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_DST;

    let masses = DeviceBuffer::from_data(&device, &objects.masses, "masses", storage_copy_dst);
    let colors = DeviceBuffer::from_data(&device, &objects.colors, "colors", storage_copy_dst);
    let shapes = DeviceBuffer::from_data(&device, &objects.shapes, "shapes", storage_copy_dst);
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
    let prioritize_compute = Arc::new(AtomicBool::new(true));

    let event_loop = EventLoop::with_user_event().build().expect("Failed to create event loop");
    let event_loop_proxy = event_loop.create_proxy();

    spawn_simulation_thread(
        device.clone(),
        queue.clone(),
        object_count,
        object_count_buffer,
        phase_state_ring_config,
        phase_state_ring.clone(),
        masses.clone(),
        exit_requested.clone(),
        prioritize_compute.clone(),
        event_loop_proxy.clone(),
    );

    let join_handle = thread::spawn({
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

    if CONFIG.headless {
        join_handle.join().unwrap();
    } else {
        let mut app = App {
            wgpu_instance,
            adapter,
            device,
            queue,
            sim_state: None,
            render_parameters: RenderParameters::default(),
            world_aabb,
            cursor_position: None,
            object_count,
            masses,
            colors,
            shapes,
            phase_state_ring_config,
            phase_state_ring,
            exit_requested,
            prioritize_compute,
            desired_maximum_frame_latency: phase_state_ring_config.n_frames,
        };
        event_loop.run_app(&mut app).expect("Failed to run app");
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
    masses: DeviceBuffer<Mass>,
    colors: DeviceBuffer<Color>,
    shapes: DeviceBuffer<Shape>,
    phase_state_ring_config: PhaseStateRingConfig,
    phase_state_ring: Arc<Mutex<PhaseStateRing>>,
    exit_requested: Arc<AtomicBool>,
    prioritize_compute: Arc<AtomicBool>,
    desired_maximum_frame_latency: usize,
}

#[derive(Copy, Clone, Debug)]
enum AppEvent {
    RedrawRequested,
}

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
        let world_size = self.world_aabb.size();
        if CONFIG.printouts {
            println!("World size: {}x{}", world_size.x, world_size.y);
        }

        let surface = self.wgpu_instance.create_surface(window.clone()).unwrap();
        let (surface_config, swapchain_format) =
            init_surface(self.desired_maximum_frame_latency, &surface, &self.adapter, &self.device, &window);

        let camera =
            DeviceBuffer::<Camera>::new(&self.device, 1, "camera", BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let shape_renderer = ShapeRenderer::new(
            &self.device,
            swapchain_format,
            camera.clone(),
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
                    let camera = orthographic_camera(view_size.cast(), world_height, &self.render_parameters);
                    state.camera.write(&self.queue, &[Camera::new(camera)]);

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

            WindowEvent::KeyboardInput { event, .. } if key_pressed(&event, KeyCode::KeyC) => {
                self.prioritize_compute.fetch_not(Ordering::SeqCst);
            }

            WindowEvent::KeyboardInput { event, .. } if key_pressed(&event, KeyCode::Escape) => event_loop.exit(),
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

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::RedrawRequested => {
                if let Some(state) = &self.sim_state {
                    state.window.request_redraw();
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.sim_state
            && !self.prioritize_compute.load(Ordering::Relaxed)
        {
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

fn init_wgpu(instance: &wgpu::Instance) -> (Adapter, Device, Queue) {
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::from_env().unwrap_or(PowerPreference::None),
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("Failed to find an appropriate adapter");

    let required_features = wgpu::Features::POLYGON_MODE_LINE | wgpu::Features::IMMEDIATES | wgpu::Features::SUBGROUP;
    let required_limits = wgpu::Limits {
        max_immediate_size: 128,
        max_storage_buffers_per_shader_stage: 16,
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
    phase_state_ring_config: PhaseStateRingConfig,
    phase_state_ring: Arc<Mutex<PhaseStateRing>>,
    masses: DeviceBuffer<Mass>,
    exit_requested: Arc<AtomicBool>,
    prioritize_compute: Arc<AtomicBool>,
    event_loop_proxy: EventLoopProxy<AppEvent>,
) {
    thread::spawn({
        let dt: f32 = CONFIG.dt;
        let dt_buffer = DeviceBuffer::from_data(&device, &[dt], "dt", BufferUsages::UNIFORM);
        let max_candidates = object_count * MAX_CANDIDATES_PER_OBJECT;
        let candidates =
            DeviceBuffer::new(&device, max_candidates, "candidates", BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let candidate_count = DeviceBuffer::new(
            &device,
            1,
            "candidate_count",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        let grid_min_x: DeviceBuffer<f32> = DeviceBuffer::new(
            &device,
            1,
            "grid min x",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );
        let grid_min_y: DeviceBuffer<f32> = DeviceBuffer::new(
            &device,
            1,
            "grid min y",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );
        let grid_max_x: DeviceBuffer<f32> = DeviceBuffer::new(
            &device,
            1,
            "grid max x",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );
        let grid_max_y: DeviceBuffer<f32> = DeviceBuffer::new(
            &device,
            1,
            "grid max y",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );
        let cell_size: DeviceBuffer<f32> = DeviceBuffer::from_data(
            &device,
            &[PARTICLE_RADIUS * 2.0],
            "cell size",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );
        let grid_size_x: DeviceBuffer<u32> = DeviceBuffer::new(
            &device,
            1,
            "grid size x",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );
        let grid_size_y: DeviceBuffer<u32> = DeviceBuffer::new(
            &device,
            1,
            "grid size y",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );
        let first_aabb: DeviceBuffer<AABB> =
            DeviceBuffer::new(&device, 1, "first aabb", BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let object_cells: DeviceBuffer<CellPosition> =
            DeviceBuffer::new(&device, object_count, "object cells", BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let max_cells = object_count * 10; // TODO: calculate properly
        let cell_object_count: DeviceBuffer<u32> = DeviceBuffer::new(
            &device,
            max_cells,
            "cell object count",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        let cell_offsets_dispatch_dimensions: DeviceBuffer<DispatchIndirectArgs> = DeviceBuffer::new(
            &device,
            N_CELL_INDIRECT_DISPATCHES,
            "cell offsets dispatch dimensions",
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        );
        let current_cell_offset = DeviceBuffer::new(
            &device,
            1,
            "current cell offset",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        let cell_offsets: DeviceBuffer<u32> =
            DeviceBuffer::new(&device, max_cells, "cell offsets", BufferUsages::STORAGE);
        let cells: DeviceBuffer<u32> =
            DeviceBuffer::new(&device, object_count * MAX_OBJECTS_PER_CELL, "cells", BufferUsages::STORAGE);

        // TODO: USE STRUCTS, too easy to mix up the parameters
        let reset_grid_aabb = ResetGridAABB::new(
            &device,
            first_aabb.clone(),
            grid_min_x.clone(),
            grid_max_x.clone(),
            grid_min_y.clone(),
            grid_max_y.clone(),
        );
        let mut calculate_grid_aabb = CalculateGridAABB::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            grid_min_x.clone(),
            grid_max_x.clone(),
            grid_min_y.clone(),
            grid_max_y.clone(),
            phase_state_ring_config,
        );
        let calculate_cell_offsets_dispatch_dimensions = CalculateCellIterationDispatchDimensions::new(
            &device,
            grid_min_x.clone(),
            grid_max_x,
            grid_min_y.clone(),
            grid_max_y,
            cell_size.clone(),
            grid_size_x.clone(),
            grid_size_y.clone(),
            cell_offsets_dispatch_dimensions.clone(),
        );
        let mut assign_object_cells = AssignObjectCells::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            grid_min_x.clone(),
            grid_min_y.clone(),
            cell_size.clone(),
            grid_size_x.clone(),
            cell_object_count.clone(),
            object_cells.clone(),
            phase_state_ring_config,
        );
        let calculate_cell_offsets = CalculateCellOffsets::new(
            &device,
            cell_offsets_dispatch_dimensions.clone(),
            grid_min_x.clone(),
            grid_min_y.clone(),
            cell_size.clone(),
            grid_size_x.clone(),
            grid_size_y.clone(),
            cell_object_count.clone(),
            current_cell_offset.clone(),
            cell_offsets.clone(),
        );
        let populate_grid_cells = PopulateGridCells::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            grid_min_x.clone(),
            grid_min_y.clone(),
            cell_size.clone(),
            grid_size_x.clone(),
            object_cells.clone(),
            cell_offsets.clone(),
            cells.clone(),
        );
        let mut broad_phase_grid = CollisionBroadPhaseGrid::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            grid_min_x.clone(),
            grid_min_y.clone(),
            cell_size.clone(),
            grid_size_x.clone(),
            grid_size_y.clone(),
            object_cells.clone(),
            cell_object_count.clone(),
            cell_offsets.clone(),
            cells.clone(),
            candidates.clone(),
            candidate_count.clone(),
            phase_state_ring_config,
        );

        let narrow_phase_dispatch_dimensions = DeviceBuffer::new(
            &device,
            1,
            "narrow phase dispatch dimensions",
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
        );
        let narrow_phase_dispatch_dimensions_calculator = NarrowPhaseDispatchIndirectArgsCalculator::new(
            &device,
            candidate_count.clone(),
            narrow_phase_dispatch_dimensions.clone(),
        );

        let collision_forces = DeviceBuffer::new(
            &device,
            object_count * 2,
            "collision forces",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let collision_reset = CollisionReset::new(&device, object_count, collision_forces.clone());
        let mut narrow_phase = NarrowPhase::new(
            &device,
            narrow_phase_dispatch_dimensions.clone(),
            candidates.clone(),
            candidate_count.clone(),
            masses.clone(),
            collision_forces.clone(),
            phase_state_ring_config,
        );
        let mut integrator = Integrator::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            dt_buffer,
            masses.clone(),
            collision_forces.clone(),
            phase_state_ring_config,
        );

        let (tx, rx) = channel::bounded(phase_state_ring_config.n_compute);
        let mut sim_step_count: usize = 0;
        let mut compute_submitted: usize = 0;
        let mut last_frame_instant = Instant::now();
        let start_instant = Instant::now();

        move || loop {
            if exit_requested.load(Ordering::Relaxed) {
                break;
            }

            if prioritize_compute.load(Ordering::Relaxed)
                && last_frame_instant.elapsed() > Duration::from_secs_f32(1.0 / 30.0)
            {
                event_loop_proxy.send_event(AppEvent::RedrawRequested).unwrap();
                last_frame_instant = Instant::now();
            }

            // Set integrator buffers, advance the 2-state sliding window that the integrator uses
            let mut phase_state_ring_guard = phase_state_ring.lock().unwrap();
            let phase_state_index = phase_state_ring_guard.current_compute_index();
            let current_phase_state = phase_state_ring_guard.current_compute().clone();
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
            encoder.clear_buffer(cell_object_count.buffer(), 0, None);
            encoder.clear_buffer(cell_offsets_dispatch_dimensions.buffer(), 0, None);
            encoder.clear_buffer(current_cell_offset.buffer(), 0, None);
            encoder.clear_buffer(candidate_count.buffer(), 0, None);

            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

            // Broad phase
            reset_grid_aabb.compute(&mut compute_pass);
            calculate_grid_aabb.compute(&mut compute_pass);
            calculate_cell_offsets_dispatch_dimensions.compute(&mut compute_pass);
            assign_object_cells.compute(&mut compute_pass);
            calculate_cell_offsets.compute(&mut compute_pass);
            populate_grid_cells.compute(&mut compute_pass);
            broad_phase_grid.compute(&mut compute_pass);

            // Narrow phase
            narrow_phase_dispatch_dimensions_calculator.compute(&mut compute_pass);
            collision_reset.compute(&mut compute_pass);
            narrow_phase.compute(&mut compute_pass);

            // Integrate
            integrator.compute(&mut compute_pass);

            drop(compute_pass);

            // Submit work
            let start = Instant::now();
            queue.submit([encoder.finish()]);
            queue.on_submitted_work_done({
                let tx = tx.clone();
                move || {
                    let _ = tx.send(());
                    if CONFIG.printouts {
                        println!("compute done in {:?}", start.elapsed());
                    }
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
            let sim_time = sim_step_count as f32 * dt;
            let real_time = start_instant.elapsed().as_secs_f32();
            if CONFIG.printouts {
                println!("Simulation rate: {} (sim {sim_time} / real {real_time})", sim_time / real_time);
            }

            // Only run for a fixed duration
            if CONFIG.sim_time_limit.is_some_and(|max_sim_time| sim_time > max_sim_time) {
                std::io::stdout().flush().unwrap();
                std::process::exit(1);
            }
        }
    });
}
