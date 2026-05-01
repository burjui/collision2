#![allow(clippy::too_many_arguments)]

pub mod aabb_renderer;
pub mod assign_object_cells;
pub mod bvh_builder;
pub mod calculate_cell_iteration_dispatch_dimensions;
pub mod calculate_cell_offsets;
pub mod calculate_grid_aabb;
pub mod calculate_grid_size;
pub mod collision_broad_phase_bvh;
pub mod collision_broad_phase_grid;
pub mod collision_forces_reset;
pub mod collision_narrow_phase;
pub mod collision_narrow_phase_dispatch_dimensions;
pub mod integration;
#[cfg(test)]
mod mock_bvh_test;
pub mod objects;
pub mod phase_state;
pub mod populate_grid_cells;
pub mod reset_cell_object_count;
pub mod reset_grid_aabb;
pub mod scene;
pub mod shaders;
pub mod shape_renderer;
pub mod typed_buffer;
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
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, CurrentSurfaceTexture, Device, DeviceDescriptor,
    InstanceDescriptor, PollType, PowerPreference, PresentMode, Queue, RenderPassColorAttachment, RenderPassDescriptor,
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
    assign_object_cells::AssignObjectCells,
    bvh_builder::{BvhBuildParameters, BvhBuilder},
    calculate_cell_iteration_dispatch_dimensions::CalculateCellIterationDispatchDimensions,
    calculate_cell_offsets::CalculateCellOffsets,
    calculate_grid_aabb::CalculateGridAABB,
    calculate_grid_size::CalculateGridSize,
    collision_broad_phase_bvh::CollisionBroadPhaseBVH,
    collision_broad_phase_grid::CollisionBroadPhaseGrid,
    collision_forces_reset::CollisionReset,
    collision_narrow_phase::NarrowPhase,
    collision_narrow_phase_dispatch_dimensions::NarrowPhaseDispatchIndirectArgsCalculator,
    integration::Integrator,
    objects::Objects,
    phase_state::PhaseStateRing,
    populate_grid_cells::PopulateGridCells,
    reset_cell_object_count::ResetCellObjectCount,
    reset_grid_aabb::ResetGridAABB,
    scene::{PARTICLE_RADIUS, create_scene},
    shaders::common::{
        AABB, BvhNode, Camera, CellPosition, DispatchIndirectArgs, GridSize, MAX_CANDIDATES_PER_OBJECT,
        MAX_OBJECTS_PER_CELL,
    },
    shape_renderer::ShapeRenderer,
    typed_buffer::TypedBuffer,
};

fn main() {
    let event_loop = EventLoop::with_user_event().build().expect("Failed to create event loop");
    let event_loop_proxy = event_loop.create_proxy();
    let mut app = App::new(event_loop_proxy);
    event_loop.run_app(&mut app).expect("Failed to run app");
}

struct App<'a> {
    render_parameters: RenderParameters,
    sim_state: Option<SimState<'a>>,
    world_aabb: AABB,
    event_loop_proxy: EventLoopProxy<AppEvent>,
    cursor_position: Option<Vector2<f32>>,
}

impl App<'_> {
    fn new(event_loop_proxy: EventLoopProxy<AppEvent>) -> Self {
        Self {
            render_parameters: RenderParameters::default(),
            world_aabb: AABB {
                min: [-3200.0, -2000.0],
                max: [3200.0, 2000.0],
            },
            sim_state: None,
            event_loop_proxy,
            cursor_position: None,
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
    object_count: usize,
    camera: TypedBuffer<Camera>,
    phase_state_ring: Arc<Mutex<PhaseStateRing>>,
    exit_requested: Arc<AtomicBool>,
    prioritize_compute: Arc<AtomicBool>,
    use_grid_broad_phase: Arc<AtomicBool>,

    window: Arc<Window>,
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,
    device: Device,
    queue: Queue,
}

impl ApplicationHandler<AppEvent> for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = create_window(event_loop);
        let wgpu_instance = wgpu::Instance::new(InstanceDescriptor::new_without_display_handle());
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

        let object_count_buffer: TypedBuffer<u32> =
            TypedBuffer::from_data(&device, &[object_count.try_into().unwrap()], "object_count", BufferUsages::UNIFORM);
        let bvh_build_params = BvhBuildParameters::new(object_count);
        // TODO: don't store leaves
        let storage_copy_dst: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_DST;

        let node_count = bvh_build_params.node_count;
        let nodes = TypedBuffer::new(&device, node_count, "nodes", storage_copy_dst);
        nodes.write_iter(&queue, (0_u32..).take(object_count).map(BvhNode::new));

        let masses = TypedBuffer::from_data(&device, &objects.masses, "masses", storage_copy_dst);
        let colors = TypedBuffer::from_data(&device, &objects.colors, "colors", storage_copy_dst);
        let shapes = TypedBuffer::from_data(&device, &objects.shapes, "shapes", storage_copy_dst);

        let phase_state_ring = Arc::new(Mutex::new(PhaseStateRing::new(
            &device,
            &queue,
            &objects.flags,
            &objects.aabbs,
            &objects.velocities,
            node_count,
        )));

        let camera = TypedBuffer::<Camera>::new(&device, 1, "camera", BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let shape_renderer =
            ShapeRenderer::new(&device, swapchain_format, camera.clone(), colors, shapes, masses.clone());
        let aabb_renderer = AabbRenderer::new(&device, swapchain_format, camera.clone(), node_count);
        let exit_requested = Arc::new(AtomicBool::new(false));
        let prioritize_compute = Arc::new(AtomicBool::new(true));
        let use_grid_broad_phase = Arc::new(AtomicBool::new(true));

        spawn_simulation_thread(
            device.clone(),
            queue.clone(),
            object_count,
            object_count_buffer,
            phase_state_ring.clone(),
            masses,
            nodes,
            bvh_build_params,
            exit_requested.clone(),
            prioritize_compute.clone(),
            use_grid_broad_phase.clone(),
            self.event_loop_proxy.clone(),
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
                }
            }
        });

        self.sim_state = Some(SimState {
            shape_renderer,
            aabb_renderer,
            object_count,
            camera,
            phase_state_ring,
            exit_requested,
            prioritize_compute,
            use_grid_broad_phase,

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
                if let Some(state) = &mut self.sim_state {
                    state.surface_config.width = size.width;
                    state.surface_config.height = size.height;
                    state.surface.configure(&state.device, &state.surface_config);
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.sim_state {
                    let view_size = state.window.inner_size();
                    let world_height = self.world_aabb.max().y - self.world_aabb.min().y;
                    let camera = orthographic_camera(view_size.cast(), world_height, &self.render_parameters);
                    state.camera.write(&state.queue, &[Camera::new(camera)]);

                    let CurrentSurfaceTexture::Success(surface_texture) = state.surface.get_current_texture() else {
                        panic!("Failed to get current surface texture");
                    };
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

            WindowEvent::KeyboardInput { event, .. } if key_pressed(&event, KeyCode::KeyC) => {
                if let Some(state) = &self.sim_state {
                    state.prioritize_compute.fetch_not(Ordering::SeqCst);
                }
            }

            WindowEvent::KeyboardInput { event, .. } if key_pressed(&event, KeyCode::KeyG) => {
                if let Some(state) = &self.sim_state {
                    state.use_grid_broad_phase.fetch_not(Ordering::SeqCst);
                }
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
        if let Some(state) = &self.sim_state {
            state.exit_requested.store(true, Ordering::SeqCst);
        }
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
            && !state.prioritize_compute.load(Ordering::Relaxed)
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

    let window_size = window.inner_size();
    let surface_config = wgpu::SurfaceConfiguration {
        present_mode: PresentMode::AutoVsync,
        desired_maximum_frame_latency: u32::try_from(PhaseStateRing::N_FRAMES).unwrap(),
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
        shape_renderer.render(&mut render_pass, range.clone());
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
    object_count: usize,
    object_count_buffer: TypedBuffer<u32>,
    phase_state_ring: Arc<Mutex<PhaseStateRing>>,
    masses: TypedBuffer<Mass>,
    nodes: TypedBuffer<BvhNode>,
    bvh_build_params: BvhBuildParameters,
    exit_requested: Arc<AtomicBool>,
    prioritize_compute: Arc<AtomicBool>,
    use_grid_broad_phase: Arc<AtomicBool>,
    event_loop_proxy: EventLoopProxy<AppEvent>,
) {
    thread::spawn({
        const DT: f32 = 0.002;
        const MAX_SIM_TIME: Option<f32> = None;

        let dt = TypedBuffer::from_data(&device, &[DT], "dt", BufferUsages::UNIFORM);
        let mut bvh_builder = BvhBuilder::new(bvh_build_params.passes, &device, nodes.clone());

        let max_candidates_per_object: usize = MAX_CANDIDATES_PER_OBJECT.try_into().unwrap();
        let max_candidates = object_count * max_candidates_per_object;
        let candidates =
            TypedBuffer::new(&device, max_candidates, "candidates", BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let candidate_count = TypedBuffer::new(
            &device,
            1,
            "candidate_count",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        let mut broad_phase_bvh = CollisionBroadPhaseBVH::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            candidates.clone(),
            candidate_count.clone(),
            nodes.clone(),
        );

        let grid_min_x: TypedBuffer<f32> = TypedBuffer::new(
            &device,
            1,
            "grid min x",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );
        let grid_min_y: TypedBuffer<f32> = TypedBuffer::new(
            &device,
            1,
            "grid min y",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );
        let grid_max_x: TypedBuffer<f32> = TypedBuffer::new(
            &device,
            1,
            "grid max x",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );
        let grid_max_y: TypedBuffer<f32> = TypedBuffer::new(
            &device,
            1,
            "grid max y",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );
        let first_aabb: TypedBuffer<AABB> =
            TypedBuffer::new(&device, 1, "first aabb", BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let cell_size: TypedBuffer<f32> = TypedBuffer::from_data(
            &device,
            &[PARTICLE_RADIUS * 2.0],
            "cell size",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );
        let grid_size: TypedBuffer<GridSize> = TypedBuffer::new(
            &device,
            1,
            "grid size",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        );
        let object_cells: TypedBuffer<CellPosition> =
            TypedBuffer::new(&device, object_count, "object cells", BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let max_cells = object_count * 10; // TODO: calculate properly
        let cell_object_count: TypedBuffer<u32> = TypedBuffer::new(
            &device,
            max_cells,
            "cell object count",
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        let cell_iteration_dispatch_dimensions: TypedBuffer<DispatchIndirectArgs> = TypedBuffer::new(
            &device,
            1,
            "cell offsets dispatch dimensions",
            BufferUsages::STORAGE | BufferUsages::INDIRECT,
        );
        let current_cell_offset = TypedBuffer::new(
            &device,
            1,
            "current cell offset",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        let cell_offsets: TypedBuffer<u32> =
            TypedBuffer::new(&device, max_cells, "cell offsets", BufferUsages::STORAGE);
        let max_objects_per_cell: usize = MAX_OBJECTS_PER_CELL.try_into().unwrap();
        let cells: TypedBuffer<u32> =
            TypedBuffer::new(&device, object_count * max_objects_per_cell, "cells", BufferUsages::STORAGE);

        let reset_grid_aabb = ResetGridAABB::new(
            &device,
            first_aabb.clone(),
            grid_min_x.clone(),
            grid_min_y.clone(),
            grid_max_x.clone(),
            grid_max_y.clone(),
        );
        let mut calculate_grid_aabb = CalculateGridAABB::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            grid_min_x.clone(),
            grid_min_y.clone(),
            grid_max_x.clone(),
            grid_max_y.clone(),
        );
        let calculate_grid_size = CalculateGridSize::new(
            &device,
            grid_min_x.clone(),
            grid_min_y.clone(),
            grid_max_x.clone(),
            grid_max_y.clone(),
            cell_size.clone(),
            grid_size.clone(),
        );
        let calculate_cell_iteration_dispatch_dimensions = CalculateCellIterationDispatchDimensions::new(
            &device,
            grid_size.clone(),
            cell_iteration_dispatch_dimensions.clone(),
        );
        let reset_cell_object_count = ResetCellObjectCount::new(
            &device,
            cell_iteration_dispatch_dimensions.clone(),
            grid_size.clone(),
            cell_object_count.clone(),
        );
        let mut assign_object_cells = AssignObjectCells::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            grid_min_x.clone(),
            grid_min_y.clone(),
            grid_size.clone(),
            cell_size.clone(),
            cell_object_count.clone(),
            object_cells.clone(),
        );
        let calculate_cell_offsets = CalculateCellOffsets::new(
            &device,
            cell_iteration_dispatch_dimensions,
            current_cell_offset.clone(),
            grid_size.clone(),
            cell_object_count.clone(),
            cell_offsets.clone(),
        );
        let populate_grid_cells = PopulateGridCells::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            grid_size.clone(),
            object_cells.clone(),
            cell_offsets.clone(),
            cells.clone(),
        );
        let mut broad_phase_grid = CollisionBroadPhaseGrid::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            grid_size.clone(),
            object_cells.clone(),
            cell_object_count.clone(),
            cell_offsets.clone(),
            cells.clone(),
            candidates.clone(),
            candidate_count.clone(),
        );

        let narrow_phase_dispatch_dimensions = TypedBuffer::new(
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

        let collision_forces = TypedBuffer::new(
            &device,
            object_count * 2,
            "collision forces",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let collision_reset = CollisionReset::new(&device, object_count.try_into().unwrap(), collision_forces.clone());
        let mut narrow_phase = NarrowPhase::new(
            &device,
            narrow_phase_dispatch_dimensions.clone(),
            candidates.clone(),
            candidate_count.clone(),
            masses.clone(),
            collision_forces.clone(),
        );
        let mut integrator = Integrator::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            dt,
            masses.clone(),
            collision_forces.clone(),
        );

        let (tx, rx) = channel::bounded(PhaseStateRing::CAPACITY);
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

            // Set up bvh and integrator buffers, advance the 2-state sliding window that the integrator uses

            let mut phase_state_ring_guard = phase_state_ring.lock().unwrap();
            let phase_state_index = phase_state_ring_guard.current_compute_index();
            let current_phase_state = phase_state_ring_guard.current_compute().clone();
            let next_phase_state = phase_state_ring_guard.next_compute().clone();
            phase_state_ring_guard.advance_compute();
            drop(phase_state_ring_guard);

            if use_grid_broad_phase.load(Ordering::Relaxed) {
                calculate_grid_aabb.prepare(&device, phase_state_index, &current_phase_state);
                assign_object_cells.prepare(&device, phase_state_index, &current_phase_state);
            } else {
                bvh_builder.prepare(&device, phase_state_index, &current_phase_state);
                broad_phase_bvh.prepare(&device, phase_state_index, &current_phase_state);
            }
            broad_phase_grid.prepare(&device, phase_state_index, &current_phase_state);
            narrow_phase.prepare(&device, phase_state_index, &current_phase_state);
            integrator.prepare(&device, phase_state_index, &current_phase_state, &next_phase_state);

            // Run the bvh builder and the integrator

            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            first_aabb.copy(0..1, current_phase_state.aabbs(), 0..1, &mut encoder);
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("bvh pass"),
                timestamp_writes: None,
            });

            reset_grid_aabb.compute(&mut compute_pass);
            calculate_grid_aabb.compute(&mut compute_pass);
            calculate_grid_size.compute(&mut compute_pass);
            calculate_cell_iteration_dispatch_dimensions.compute(&mut compute_pass);
            reset_cell_object_count.compute(&mut compute_pass);
            assign_object_cells.compute(&mut compute_pass);
            current_cell_offset.write(&queue, &[0]);
            calculate_cell_offsets.compute(&mut compute_pass);
            populate_grid_cells.compute(&mut compute_pass);

            candidate_count.write(&queue, &[0]);
            if use_grid_broad_phase.load(Ordering::Relaxed) {
                broad_phase_grid.compute(&mut compute_pass);
            } else {
                bvh_builder.compute(&mut compute_pass);
                broad_phase_bvh.compute(&queue, &mut compute_pass);
            }

            narrow_phase_dispatch_dimensions_calculator.compute(&mut compute_pass);
            collision_reset.compute(&mut compute_pass);
            narrow_phase.compute(&mut compute_pass);
            integrator.compute(&mut compute_pass);
            drop(compute_pass);

            // Submit work

            let start = Instant::now();
            queue.submit([encoder.finish()]);
            queue.on_submitted_work_done({
                let tx = tx.clone();
                move || {
                    let _ = tx.send(());
                    println!("compute done in {:?}", start.elapsed());
                }
            });

            // Max PhaseStateRing::N_COMPUTE - 1 integrations at a time

            compute_submitted += 1;
            if compute_submitted >= PhaseStateRing::N_COMPUTE - 1 {
                rx.recv().unwrap();
                compute_submitted -= 1;
            }

            // Print stats

            sim_step_count += 1;
            let sim_time = sim_step_count as f32 * DT;
            let real_time = start_instant.elapsed().as_secs_f32();
            println!("Simulation rate: {} (sim {sim_time} / real {real_time})", sim_time / real_time);

            // Only run for a fixed duration

            if MAX_SIM_TIME.is_some_and(|max_sim_time| sim_time > max_sim_time) {
                std::io::stdout().flush().unwrap();
                std::process::exit(1);
            }
        }
    });
}
