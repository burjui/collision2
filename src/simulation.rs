use std::{
    io::{Write as _, stdout},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossbeam::channel;
use wgpu::{CommandEncoderDescriptor, ComputePass, ComputePassDescriptor, Device, Queue};
use winit::event_loop::EventLoopProxy;

use crate::{
    assign_object_cells::{AssignObjectCells, AssignObjectCellsParams},
    calculate_cell_offsets::{CalculateCellOffsets, CalculateCellOffsetsParams},
    calculate_cell_offsets_dispatch_dimensions::{
        CellIterationDispatchDimensions, CellIterationDispatchDimensionsParams,
    },
    calculate_grid_aabb::{CalculateGridAABB, CalculateGridAABBParams},
    collisions::{Collisions, CollisionsParams},
    command_timings::CommandTimings,
    config::AppConfig,
    cursor_impulse::{CursorImpulse, CursorImpulseParams},
    device_buffer::DeviceBuffer,
    integrator::Integrator,
    phase_state::{PhaseStateRing, PhaseStateRingConfig},
    populate_grid_cells::{PopulateGridCells, PopulateGridCellsParams},
    recorder::Recorder,
    reset_grid_aabb::{ResetGridAABB, ResetGridAABBParams},
    shaders::{
        calculate_cell_offsets_dispatch_dimensions::N_CELL_INDIRECT_DISPATCHES,
        common::{AABB, MAX_OBJECTS_PER_CELL, Mass},
    },
    util::{COPY_DST, COPY_SRC, ComputeStage, INDIRECT, STORAGE, UNIFORM, map_mutex},
};

#[derive(Copy, Clone, Debug)]
pub enum SimulationRequest {
    RedrawRequested,
    ExitEventLoop,
}

pub fn spawn_simulation_thread(
    app_config: Arc<AppConfig>,
    device: Device,
    queue: Queue,
    object_count: u32,
    particle_radius: DeviceBuffer<f32>,
    world_aabb: AABB,
    phase_state_ring_config: PhaseStateRingConfig,
    phase_state_ring: Arc<Mutex<PhaseStateRing>>,
    masses: DeviceBuffer<Mass>,
    cursor_position: DeviceBuffer<[f32; 2]>,
    cursor_impulse_radius: DeviceBuffer<f32>,
    cursor_impulse_speed: DeviceBuffer<f32>,
    exit_requested: Arc<AtomicBool>,
    event_loop_proxy: Option<EventLoopProxy<SimulationRequest>>,
    pause_simulation: Arc<AtomicBool>,
    surface_initialized: Arc<AtomicBool>,
    mut recorder: Option<Recorder>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        if !surface_initialized.load(Ordering::SeqCst) {
            println!("Waiting for surface initialization");
        }
        while !surface_initialized.load(Ordering::SeqCst) {
            thread::yield_now();
        }

        println!("Starting simulation thread");

        let particle_radius = &particle_radius;
        let object_count_buffer = &DeviceBuffer::from_data(&device, &[object_count], "object_count", UNIFORM);
        let dt: f32 = app_config.dt;
        let dt_buffer = DeviceBuffer::from_data(&device, &[dt], "dt", UNIFORM);
        let grid_min_x = &DeviceBuffer::new(&device, 1, "grid min x", STORAGE | UNIFORM | COPY_SRC);
        let grid_min_y = &DeviceBuffer::new(&device, 1, "grid min y", STORAGE | UNIFORM | COPY_SRC);
        let grid_max_x = &DeviceBuffer::new(&device, 1, "grid max x", STORAGE | UNIFORM | COPY_SRC);
        let grid_max_y = &DeviceBuffer::new(&device, 1, "grid max y", STORAGE | UNIFORM | COPY_SRC);
        let grid_size_x = &DeviceBuffer::new(&device, 1, "grid size x", STORAGE | UNIFORM | COPY_SRC);
        let grid_size_y = &DeviceBuffer::new(&device, 1, "grid size y", STORAGE | UNIFORM | COPY_SRC);
        let object_cells = &DeviceBuffer::new(&device, object_count, "object cells", STORAGE | COPY_SRC);
        let cell_size = app_config.particle_radius * 2.0;
        let grid_size = world_aabb.size() / cell_size;
        let grid_cells_x = grid_size.x.ceil() as u32;
        let grid_cells_y = grid_size.y.ceil() as u32;
        let max_cells = grid_cells_x * grid_cells_y;
        let cell_object_count =
            &DeviceBuffer::new(&device, max_cells, "cell object count", STORAGE | UNIFORM | COPY_DST | COPY_SRC);
        let dispatch_dimensions = &DeviceBuffer::new(
            &device,
            N_CELL_INDIRECT_DISPATCHES,
            "cell offsets dispatch dimensions",
            STORAGE | INDIRECT | COPY_DST,
        );
        let current_cell_offset = &DeviceBuffer::new(&device, 1, "current cell offset", STORAGE | COPY_SRC | COPY_DST);
        let cell_offsets = &DeviceBuffer::new(&device, max_cells, "cell offsets", STORAGE);
        let cells = &DeviceBuffer::new(&device, object_count * MAX_OBJECTS_PER_CELL, "cells", STORAGE);
        let forces = &DeviceBuffer::new(&device, object_count * 2, "collision forces", STORAGE | COPY_SRC | COPY_DST);
        let stiffness = &DeviceBuffer::from_data(&device, &[app_config.stiffness], "stiffness", UNIFORM | COPY_SRC);
        let gamma_coeff: f32 =
            (3.0 / 2.0) * (1.0 - app_config.restitution.powi(2)) / 5.0_f32.sqrt() * app_config.stiffness.sqrt();
        let gamma_coeff = &DeviceBuffer::from_data(&device, &[gamma_coeff], "gamma_coeff", UNIFORM);
        let effective_mass = app_config.particle_mass.powi(2) / (app_config.particle_mass * 2.0);

        // NOTE: This is only valid while all particles have the same mass, which is the case at the moment.
        // As soon as they become different, this term has to be calculated in the shader,
        // e.g. if I simulate fission.
        let sqrt_m_eff = &DeviceBuffer::from_data(&device, &[effective_mass.sqrt()], "sqrt_m_eff", UNIFORM);

        let safety_margin = app_config.particle_radius * 2.0;
        let constraints = AABB {
            min: [world_aabb.min[0] + safety_margin, world_aabb.min[1] + safety_margin],
            max: [world_aabb.max[0] - safety_margin, world_aabb.max[1] - safety_margin],
        };
        let constraints_buffer = &DeviceBuffer::from_data(&device, &[constraints], "constraints", UNIFORM | COPY_SRC);

        let mut cursor_impulse = CursorImpulse::new(
            &device,
            CursorImpulseParams {
                phase_state_ring_config,
                object_count,
                object_count_buffer: object_count_buffer.clone(),
                cursor_position: cursor_position.clone(),
                cursor_impulse_radius: cursor_impulse_radius.clone(),
                cursor_impulse_speed: cursor_impulse_speed.clone(),
            },
        );
        let reset_grid_aabb = ResetGridAABB::new(
            &device,
            ResetGridAABBParams {
                grid_min_x,
                grid_max_x,
                grid_min_y,
                grid_max_y,
            },
        );
        let mut calculate_grid_aabb = CalculateGridAABB::new(
            &device,
            CalculateGridAABBParams {
                phase_state_ring_config,
                object_count,
                object_count_buffer,
                particle_radius,
                grid_min_x,
                grid_max_x,
                grid_min_y,
                grid_max_y,
            },
        );
        let cell_offsets_dispatch_dimensions = CellIterationDispatchDimensions::new(
            &device,
            CellIterationDispatchDimensionsParams {
                dispatch_dimensions,
                particle_radius,
                grid_min_x,
                grid_max_x,
                grid_min_y,
                grid_max_y,
                grid_size_x,
                grid_size_y,
            },
        );
        let mut assign_object_cells = AssignObjectCells::new(
            &device,
            AssignObjectCellsParams {
                phase_state_ring_config,
                object_count,
                object_count_buffer,
                particle_radius,
                grid_min_x,
                grid_min_y,
                grid_size_x,
                grid_size_y,
                cell_object_count,
                object_cells,
            },
        );
        let calculate_cell_offsets = CalculateCellOffsets::new(
            &device,
            CalculateCellOffsetsParams {
                dispatch_dimensions,
                grid_min_x,
                grid_min_y,
                grid_size_x,
                grid_size_y,
                cell_object_count,
                current_cell_offset,
                cell_offsets,
            },
        );
        let populate_grid_cells = PopulateGridCells::new(
            &device,
            PopulateGridCellsParams {
                object_count,
                object_count_buffer,
                grid_size_x,
                object_cells,
                cell_offsets,
                cells,
            },
        );
        let mut collisions = Collisions::new(
            &device,
            CollisionsParams {
                phase_state_ring_config,
                object_count,
                object_count_buffer,
                particle_radius,
                grid_min_x,
                grid_min_y,
                grid_size_x,
                grid_size_y,
                object_cells,
                cell_object_count,
                cell_offsets,
                cells,
                forces,
                stiffness,
                gamma_coeff,
                sqrt_m_eff,
            },
        );
        let mut integrator = Integrator::new(
            &app_config,
            &device,
            object_count,
            object_count_buffer.clone(),
            particle_radius.clone(),
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

        loop {
            if exit_requested.load(Ordering::Relaxed) {
                break;
            }

            let sim_time = sim_step_count as f32 * app_config.dt;
            let real_time = start_instant.elapsed().as_secs_f32();
            let print_sim_rate =
                move || println!("Simulation rate: {} (sim {sim_time} / real {real_time})", sim_time / real_time);

            if app_config.sim_time_limit > 0.0 && sim_time >= app_config.sim_time_limit {
                println!("Simulation time limit reached");
                print_sim_rate();
                stdout().flush().unwrap();
                exit_requested.store(true, Ordering::SeqCst);
                if let Some(event_loop_proxy) = &event_loop_proxy
                    && app_config.exit_at_limit
                {
                    let _ = event_loop_proxy.send_event(SimulationRequest::ExitEventLoop);
                }
                break;
            }

            if last_frame_instant.elapsed() >= Duration::from_secs_f32(1.0 / app_config.fps) {
                last_frame_instant = Instant::now();
                map_mutex(&phase_state_ring, PhaseStateRing::advance_frame);

                if let Some(event_loop_proxy) = &event_loop_proxy {
                    let _ = event_loop_proxy.send_event(SimulationRequest::RedrawRequested);
                }
            }

            if pause_simulation.load(Ordering::Relaxed) {
                thread::yield_now();
                continue;
            }

            // Set integrator buffers, advance the 2-state sliding window that the integrator uses
            let (mut current_phase_state, phase_state_index, next_phase_state, _) =
                map_mutex(&phase_state_ring, |phase_state_ring| {
                    (
                        phase_state_ring.current_compute().clone(),
                        phase_state_ring.current_compute_index(),
                        phase_state_ring.next_compute().clone(),
                        phase_state_ring.advance_compute(),
                    )
                });

            cursor_impulse.prepare(&device, phase_state_index, &current_phase_state);
            calculate_grid_aabb.prepare(&device, phase_state_index, &current_phase_state);
            assign_object_cells.prepare(&device, phase_state_index, &current_phase_state);
            collisions.prepare(&device, phase_state_index, &current_phase_state);
            integrator.prepare(&device, phase_state_index, &current_phase_state, &next_phase_state);

            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());

            let timings = current_phase_state.command_timings();
            timings.measure(&mut encoder, "Clear buffers", |encoder| {
                encoder.clear_buffer(cell_object_count.buffer(), 0, None);
                encoder.clear_buffer(dispatch_dimensions.buffer(), 0, None);
                encoder.clear_buffer(current_cell_offset.buffer(), 0, None);
                encoder.clear_buffer(forces.buffer(), 0, None);
            });

            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            compute(&cursor_impulse, &mut compute_pass, timings);
            compute(&reset_grid_aabb, &mut compute_pass, timings);
            compute(&calculate_grid_aabb, &mut compute_pass, timings);
            compute(&cell_offsets_dispatch_dimensions, &mut compute_pass, timings);
            compute(&assign_object_cells, &mut compute_pass, timings);
            compute(&calculate_cell_offsets, &mut compute_pass, timings);
            compute(&populate_grid_cells, &mut compute_pass, timings);
            compute(&collisions, &mut compute_pass, timings);
            // compute(&narrow_phase_dispatch_dimensions, &mut compute_pass, timings);
            // compute(&narrow_phase, &mut compute_pass, timings);
            compute(&integrator, &mut compute_pass, timings);
            drop(compute_pass);

            encoder.clear_buffer(cursor_impulse_speed.buffer(), 0, None);

            let timings_reader = timings.resolve(&mut encoder, timestamp_period);

            // Submit work
            let start = Instant::now();
            queue.submit([encoder.finish()]);
            queue.on_submitted_work_done({
                let tx = tx.clone();
                let enable_printouts = app_config.printouts;
                move || {
                    if enable_printouts {
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

            sim_step_count += 1;

            if let Some(recorder) = &mut recorder {
                recorder.record_frame(sim_step_count);
            }
        }
    })
}

fn compute<T: ComputeStage>(stage: &T, compute_pass: &mut ComputePass, timings: &mut CommandTimings) {
    timings.measure_compute(compute_pass, T::LABEL, |compute_pass| stage.compute_impl(compute_pass));
}
