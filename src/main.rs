#![allow(clippy::too_many_arguments)]

pub mod aabb_renderer;
pub mod assign_object_cells;
pub mod calculate_cell_offsets;
pub mod calculate_cell_offsets_dispatch_dimensions;
pub mod calculate_grid_aabb;
pub mod camera;
pub mod collisions;
pub mod command_timings;
pub mod compute_stage;
pub mod config;
pub mod device_buffer;
pub mod integrator;
pub mod objects;
pub mod phase_state;
pub mod populate_grid_cells;
pub mod recorder;
pub mod renderer;
pub mod reset_grid_aabb;
pub mod scene;

#[allow(clippy::missing_safety_doc)]
pub mod shaders;

pub mod cursor_impulse;
pub mod shape_renderer;
pub mod simulation;
pub mod util;

use core::panic;
use std::{
    env::args,
    io::{Write as _, stdout},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self},
};

use nalgebra::Vector2;
use pollster::block_on;
use renderer::RenderParameters;
use shaders::common::Mass;
use threadpool::ThreadPool;
use wgpu::{
    Adapter, CurrentSurfaceTexture, Device, DeviceDescriptor, InstanceDescriptor, PollType, PresentMode, Queue,
    Surface, SurfaceConfiguration, TextureFormat, TextureViewDescriptor, util::initialize_adapter_from_env_or_default,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::KeyCode,
    window::{Fullscreen, Window, WindowAttributes, WindowId},
};

use crate::{
    aabb_renderer::AabbRenderer,
    camera::orthographic_camera,
    config::AppConfig,
    device_buffer::DeviceBuffer,
    objects::Objects,
    phase_state::{PhaseStateRing, PhaseStateRingConfig},
    recorder::{Recorder, RecorderParams},
    renderer::render_scene,
    scene::create_scene,
    shaders::common::{AABB, Camera, Color, Shape},
    shape_renderer::ShapeRenderer,
    simulation::{SimulationRequest, spawn_simulation_thread},
    util::{COPY_DST, COPY_SRC, STORAGE, UNIFORM},
};

fn main() {
    let path = args().nth(1);
    let app_config = Arc::new(AppConfig::new(path.as_deref()).unwrap());
    if app_config.print_config {
        println!("{:#?}", *app_config);
    }

    let wgpu_instance = wgpu::Instance::new(InstanceDescriptor::new_without_display_handle());
    let (adapter, device, queue) = init_wgpu(&wgpu_instance);
    if app_config.print_adapter {
        println!("{:#?}", adapter.get_info());
    }

    let mut objects = Objects::default();
    // Dimensions have to be positive, refer to calculate_grid_aabb.wgsl
    let world_aabb = AABB {
        min: [0.0, 0.0],
        max: app_config.world_size(),
    };
    create_scene(&app_config, &mut objects, world_aabb);

    let object_count: u32 = objects.flags.len().try_into().unwrap();
    println!("Object count: {}", object_count);

    stdout().flush().unwrap();

    let spectrum_width =
        DeviceBuffer::from_data(&device, &[app_config.spectrum_width], "energy spectrum width", UNIFORM);
    let masses = DeviceBuffer::from_data(&device, &objects.masses, "masses", STORAGE | COPY_DST);
    let colors = DeviceBuffer::from_data(&device, &objects.colors, "colors", STORAGE | COPY_DST);
    let shapes = DeviceBuffer::from_data(&device, &objects.shapes, "shapes", STORAGE | COPY_DST);
    let phase_state_ring_config = PhaseStateRingConfig {
        n_frames: app_config.n_frames,
        n_compute: app_config.n_compute,
    };
    let phase_state_ring = Arc::new(Mutex::new(PhaseStateRing::new(
        phase_state_ring_config,
        &app_config,
        &device,
        object_count,
        &objects.flags,
        &objects.positions,
        &objects.velocities,
    )));

    let exit_requested = Arc::new(AtomicBool::new(false));
    let render_parameters = RenderParameters {
        offset: Vector2::new(world_aabb.size().x * 0.5, -world_aabb.size().y * 0.5),
        ..Default::default()
    };
    let pause_simulation = Arc::new(AtomicBool::new(false));
    let particle_radius = DeviceBuffer::from_data(&device, &[app_config.particle_radius], "particle radius", UNIFORM);
    let surface_initialized = Arc::new(AtomicBool::new(app_config.headless));
    let event_loop = if app_config.headless {
        None
    } else {
        Some(EventLoop::with_user_event().build().expect("Failed to create event loop"))
    };
    let event_loop_proxy = event_loop.as_ref().map(|event_loop| event_loop.create_proxy());
    let cursor_position_buffer =
        DeviceBuffer::from_data(&device, &[[0.0, 0.0]], "kick center", UNIFORM | COPY_SRC | COPY_DST);
    let cursor_impulse_radius = DeviceBuffer::from_data(
        &device,
        &[app_config.cursor_impulse_radius],
        "kick radius",
        UNIFORM | COPY_SRC | COPY_DST,
    );
    let cursor_impulse_speed =
        DeviceBuffer::from_data(&device, &[0.0], "kick magnitude", UNIFORM | COPY_SRC | COPY_DST);
    let mut recorder = None;
    if !app_config.output_path.is_empty() {
        recorder = Some(Recorder::new(RecorderParams {
            app_config: app_config.clone(),
            device: device.clone(),
            queue: queue.clone(),
            output_path: app_config.output_path.clone(),
            world_aabb,
            render_parameters,
            particle_radius: particle_radius.clone(),
            spectrum_width: spectrum_width.clone(),
            colors: colors.clone(),
            shapes: shapes.clone(),
            masses: masses.clone(),
            phase_state_ring: phase_state_ring.clone(),
            phase_state_ring_config,
            object_count,
        }));
    }
    let sim_join_handle = spawn_simulation_thread(
        app_config.clone(),
        device.clone(),
        queue.clone(),
        object_count,
        particle_radius.clone(),
        world_aabb,
        phase_state_ring_config,
        phase_state_ring.clone(),
        masses.clone(),
        cursor_position_buffer.clone(),
        cursor_impulse_radius.clone(),
        cursor_impulse_speed.clone(),
        exit_requested.clone(),
        event_loop_proxy,
        pause_simulation.clone(),
        surface_initialized.clone(),
        recorder,
    );

    let poll_join_handle = thread::spawn({
        let device = device.clone();
        let exit_requested = exit_requested.clone();
        move || {
            let pool = ThreadPool::new(num_cpus::get());
            loop {
                let device = device.clone();
                if pool.queued_count() < pool.max_count() {
                    pool.execute(move || {
                        let _ = device.poll(PollType::Poll).unwrap();
                    });
                }
                if exit_requested.load(Ordering::Relaxed) {
                    break;
                }
                std::thread::yield_now();
            }
        }
    });

    if app_config.headless {
        println!("Running in headless mode");
        sim_join_handle.join().unwrap();
        poll_join_handle.join().unwrap();
    } else {
        println!("Running in windowed mode");
        let mut app = App {
            app_config: app_config.clone(),
            wgpu_instance,
            adapter,
            device,
            queue,
            sim_state: None,
            render_parameters,
            world_aabb,
            cursor_position: None,
            object_count,
            particle_radius,
            spectrum_width: spectrum_width.clone(),
            colors,
            shapes,
            masses,
            cursor_position_buffer,
            cursor_impulse_speed,
            phase_state_ring_config,
            phase_state_ring,
            exit_requested,
            desired_maximum_frame_latency: phase_state_ring_config.n_frames,
            pause_simulation: pause_simulation.clone(),
            surface_initialized,
            lmb_down: false,
        };
        event_loop.unwrap().run_app(&mut app).expect("Failed to run app");
    }
}

struct App<'a> {
    app_config: Arc<AppConfig>,
    wgpu_instance: wgpu::Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    sim_state: Option<SimState<'a>>,
    render_parameters: RenderParameters,
    world_aabb: AABB,
    cursor_position: Option<Vector2<f32>>,
    object_count: u32,
    particle_radius: DeviceBuffer<f32>,
    spectrum_width: DeviceBuffer<f32>,
    colors: DeviceBuffer<Color>,
    shapes: DeviceBuffer<Shape>,
    masses: DeviceBuffer<Mass>,
    cursor_position_buffer: DeviceBuffer<[f32; 2]>,
    cursor_impulse_speed: DeviceBuffer<f32>,
    phase_state_ring_config: PhaseStateRingConfig,
    phase_state_ring: Arc<Mutex<PhaseStateRing>>,
    exit_requested: Arc<AtomicBool>,
    desired_maximum_frame_latency: usize,
    pause_simulation: Arc<AtomicBool>,
    surface_initialized: Arc<AtomicBool>,
    lmb_down: bool,
}

struct SimState<'a> {
    shape_renderer: ShapeRenderer,
    aabb_renderer: AabbRenderer,
    camera: DeviceBuffer<Camera>,
    window: Arc<Window>,
    surface: Surface<'a>,
    surface_config: SurfaceConfiguration,
}

impl ApplicationHandler<SimulationRequest> for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        println!("Initializing window");
        let window = create_window(event_loop);
        let window_size = window.inner_size();
        if self.app_config.printouts {
            println!("Window size: {}x{}", window_size.width, window_size.height);
        }

        println!("Initializing surface");

        let surface = self.wgpu_instance.create_surface(window.clone()).unwrap();
        let (surface_config, swapchain_format) =
            init_surface(self.desired_maximum_frame_latency, &surface, &self.adapter, &self.device, &window);
        let camera = DeviceBuffer::<Camera>::new(&self.device, 1, "camera", UNIFORM | COPY_DST);
        let shape_renderer = ShapeRenderer::new(
            &self.device,
            swapchain_format,
            camera.clone(),
            self.particle_radius.clone(),
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
            self.particle_radius.clone(),
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
                println!("Resizing window to {}x{}", size.width, size.height);
                if let Some(state) = &mut self.sim_state {
                    state.surface_config.width = size.width;
                    state.surface_config.height = size.height;
                    let was_paused = self.pause_simulation.swap(true, Ordering::SeqCst);
                    self.device.poll(PollType::wait_indefinitely()).unwrap();
                    state.surface.configure(&self.device, &state.surface_config);
                    self.device.poll(PollType::wait_indefinitely()).unwrap();
                    self.pause_simulation.store(was_paused, Ordering::SeqCst);
                    self.surface_initialized.store(true, Ordering::SeqCst);
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.sim_state
                    && !self.pause_simulation.load(Ordering::Relaxed)
                {
                    let view_size = state.window.inner_size();
                    let world_height = self.world_aabb.max().y - self.world_aabb.min().y;
                    let camera_matrix = orthographic_camera(
                        view_size.cast(),
                        world_height,
                        self.render_parameters.zoom,
                        self.render_parameters.offset,
                    );
                    state.camera.write(&self.queue, &[Camera::new(camera_matrix)]);
                    self.device.poll(PollType::wait_indefinitely()).unwrap();

                    let Some(surface_texture) = acquire_surface_texture(
                        self.wgpu_instance.clone(),
                        self.device.clone(),
                        &mut state.surface,
                        state.surface_config.clone(),
                        state.window.clone(),
                        self.pause_simulation.clone(),
                    ) else {
                        panic!("Failed to get current surface texture");
                    };
                    let surface_texture_view = surface_texture.texture.create_view(&TextureViewDescriptor::default());
                    render_scene(
                        &self.app_config,
                        &self.device,
                        &self.queue,
                        surface_texture_view,
                        &self.render_parameters,
                        &mut state.shape_renderer,
                        &mut state.aabb_renderer,
                        &self.phase_state_ring,
                        0..self.object_count,
                        |_| {},
                        "Rendering to screen",
                    );
                    state.window.pre_present_notify();
                    self.queue.present(surface_texture);
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
                if let Some(state) = &self.sim_state
                    && let Some(cursor_position) = self.cursor_position
                {
                    let zoom_old = self.render_parameters.zoom;
                    let zoom_new = self.render_parameters.zoom * (1.0 + dy * 0.1);
                    let view_size = state.window.inner_size();
                    let view_center = Vector2::new(view_size.width as f32 / 2.0, view_size.height as f32 / 2.0);
                    let cursor_relative_to_center = cursor_position - view_center;
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
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let position: [f64; 2] = position.into();
                self.cursor_position = Some(Vector2::from(position).cast());
            }

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => match state {
                ElementState::Pressed => {
                    self.lmb_down = true;
                    self.kick_at_cursor();
                }
                ElementState::Released => self.lmb_down = false,
            },

            _ => (),
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.exit_requested.store(true, Ordering::SeqCst);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: SimulationRequest) {
        match event {
            SimulationRequest::RedrawRequested => {
                if let Some(state) = &self.sim_state {
                    state.window.request_redraw();
                }
            }
            SimulationRequest::ExitEventLoop => event_loop.exit(),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if self.lmb_down {
            self.kick_at_cursor();
        }
    }
}

impl App<'_> {
    fn kick_at_cursor(&self) {
        let state = self.sim_state.as_ref().unwrap();
        let cursor_position = mouse_to_world_coords(
            self.cursor_position.unwrap(),
            self.world_aabb,
            state.window.inner_size(),
            self.render_parameters.offset,
            self.render_parameters.zoom,
        );
        self.cursor_position_buffer.write(&self.queue, &[cursor_position.into()]);
        self.cursor_impulse_speed.write(&self.queue, &[100.0]);
    }
}

fn acquire_surface_texture(
    wgpu_instance: wgpu::Instance,
    device: Device,
    surface: &mut Surface,
    surface_config: SurfaceConfiguration,
    window: Arc<Window>,
    pause_simulation: Arc<AtomicBool>,
) -> Option<wgpu::SurfaceTexture> {
    match surface.get_current_texture() {
        CurrentSurfaceTexture::Success(frame) => Some(frame),
        // If we timed out or the window is occluded, skip this frame:
        CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => None,
        // If the surface is outdated or suboptimal, reconfigure and retry.
        CurrentSurfaceTexture::Suboptimal(texture) => {
            drop(texture);
            let was_paused = pause_simulation.swap(true, Ordering::SeqCst);
            device.poll(PollType::wait_indefinitely()).unwrap();
            surface.configure(&device, &surface_config);
            device.poll(PollType::wait_indefinitely()).unwrap();
            pause_simulation.store(was_paused, Ordering::SeqCst);
            match surface.get_current_texture() {
                CurrentSurfaceTexture::Success(frame) | CurrentSurfaceTexture::Suboptimal(frame) => Some(frame),
                other => panic!("Failed to acquire next surface texture: {other:?}"),
            }
        }
        CurrentSurfaceTexture::Outdated => {
            let was_paused = pause_simulation.swap(true, Ordering::SeqCst);
            device.poll(PollType::wait_indefinitely()).unwrap();
            surface.configure(&device, &surface_config);
            device.poll(PollType::wait_indefinitely()).unwrap();
            pause_simulation.store(was_paused, Ordering::SeqCst);
            match surface.get_current_texture() {
                CurrentSurfaceTexture::Success(frame) | CurrentSurfaceTexture::Suboptimal(frame) => Some(frame),
                other => panic!("Failed to acquire next surface texture: {other:?}"),
            }
        }
        CurrentSurfaceTexture::Validation => {
            unreachable!("No error scope registered, so validation errors will panic")
        }
        // If the surface is lost, recreate and reconfigure it.
        CurrentSurfaceTexture::Lost => {
            *surface = wgpu_instance.create_surface(window).unwrap();
            let was_paused = pause_simulation.swap(true, Ordering::SeqCst);
            device.poll(PollType::wait_indefinitely()).unwrap();
            surface.configure(&device, &surface_config);
            device.poll(PollType::wait_indefinitely()).unwrap();
            pause_simulation.store(was_paused, Ordering::SeqCst);
            match surface.get_current_texture() {
                CurrentSurfaceTexture::Success(frame) | CurrentSurfaceTexture::Suboptimal(frame) => Some(frame),
                other => panic!("Failed to acquire next surface texture: {other:?}"),
            }
        }
    }
}

fn mouse_to_world_coords(
    cursor_position: Vector2<f32>,
    world_aabb: AABB,
    view_size: PhysicalSize<u32>,
    offset: Vector2<f32>,
    zoom: f32,
) -> Vector2<f32> {
    let view_width = view_size.width as f32;
    let view_height = view_size.height as f32;
    let world_height = world_aabb.max().y - world_aabb.min().y;
    let world_width = world_height * view_width / view_height;
    let x = (cursor_position.x / view_width - 0.5) * world_width;
    let y = (0.5 - cursor_position.y / view_height) * world_height;
    Vector2::new((x + offset.x) / zoom, (y - offset.y) / zoom)
}

fn key_pressed(event: &KeyEvent, key: KeyCode) -> bool {
    event.state == ElementState::Pressed && event.physical_key == key
}

fn create_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    let mut window_attributes = WindowAttributes::default();
    window_attributes.inner_size = Some(PhysicalSize::new(1920, 1080).into());
    window_attributes.fullscreen = Some(Fullscreen::Borderless(None));
    let window = event_loop.create_window(window_attributes).expect("Failed to create window");
    Arc::new(window)
}

fn init_wgpu(instance: &wgpu::Instance) -> (Adapter, Device, Queue) {
    let adapter = block_on(initialize_adapter_from_env_or_default(instance, None))
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
