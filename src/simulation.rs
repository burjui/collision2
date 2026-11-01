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
    assign_object_cells::AssignObjectCells,
    buffer_sets::BroadPhaseBuffers,
    calculate_cell_offsets::CalculateCellOffsets,
    calculate_cell_offsets_dispatch_dimensions::CellIterationDispatchDimensions,
    calculate_grid_aabb::CalculateGridAABB,
    collision_broad_phase_grid::CollisionBroadPhaseGrid,
    collision_narrow_phase::NarrowPhase,
    collision_narrow_phase_dispatch_dimensions::NarrowPhaseDispatchIndirectArgs,
    command_timings::CommandTimings,
    config::AppConfig,
    device_buffer::DeviceBuffer,
    integrator::Integrator,
    phase_state::{PhaseStateRing, PhaseStateRingConfig},
    populate_grid_cells::PopulateGridCells,
    recorder::Recorder,
    reset_grid_aabb::ResetGridAABB,
    shaders::{
        calculate_cell_offsets_dispatch_dimensions::N_CELL_INDIRECT_DISPATCHES,
        common::{AABB, MAX_CANDIDATES_PER_OBJECT, MAX_OBJECTS_PER_CELL, Mass},
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
    kick_center: DeviceBuffer<[f32; 2]>,
    kick_radius: DeviceBuffer<f32>,
    kick_magnitude: DeviceBuffer<f32>,
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

        let object_count_buffer = DeviceBuffer::from_data(&device, &[object_count], "object_count", UNIFORM);
        let dt: f32 = app_config.dt;
        let dt_buffer = DeviceBuffer::from_data(&device, &[dt], "dt", UNIFORM);
        let max_candidates = object_count * MAX_CANDIDATES_PER_OBJECT;
        let candidates = DeviceBuffer::new(&device, max_candidates, "candidates", STORAGE | COPY_SRC);
        let candidate_count = DeviceBuffer::new(&device, 1, "candidate_count", STORAGE | UNIFORM | COPY_DST | COPY_SRC);
        let grid_min_x = DeviceBuffer::new(&device, 1, "grid min x", STORAGE | UNIFORM | COPY_SRC);
        let grid_min_y = DeviceBuffer::new(&device, 1, "grid min y", STORAGE | UNIFORM | COPY_SRC);
        let grid_max_x = DeviceBuffer::new(&device, 1, "grid max x", STORAGE | UNIFORM | COPY_SRC);
        let grid_max_y = DeviceBuffer::new(&device, 1, "grid max y", STORAGE | UNIFORM | COPY_SRC);
        let grid_size_x = DeviceBuffer::new(&device, 1, "grid size x", STORAGE | UNIFORM | COPY_SRC);
        let grid_size_y = DeviceBuffer::new(&device, 1, "grid size y", STORAGE | UNIFORM | COPY_SRC);
        let object_cells = DeviceBuffer::new(&device, object_count, "object cells", STORAGE | COPY_SRC);
        let cell_size = app_config.particle_radius * 2.0;
        let grid_size = world_aabb.size() / cell_size;
        let grid_cells_x = grid_size.x.ceil() as u32;
        let grid_cells_y = grid_size.y.ceil() as u32;
        let max_cells = grid_cells_x * grid_cells_y;
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
        let stiffness = DeviceBuffer::from_data(&device, &[app_config.stiffness], "stiffness", UNIFORM | COPY_SRC);
        let restitution =
            DeviceBuffer::from_data(&device, &[app_config.restitution], "restitution", UNIFORM | COPY_SRC);
        let safety_margin = app_config.particle_radius * 2.0;
        let constraints = AABB {
            min: [world_aabb.min[0] + safety_margin, world_aabb.min[1] + safety_margin],
            max: [world_aabb.max[0] - safety_margin, world_aabb.max[1] - safety_margin],
        };
        let constraints_buffer = DeviceBuffer::from_data(&device, &[constraints], "constraints", UNIFORM | COPY_SRC);
        let broad_phase_buffers = BroadPhaseBuffers {
            particle_radius: particle_radius.clone(),
            grid_min_x: grid_min_x.clone(),
            grid_max_x: grid_max_x.clone(),
            grid_min_y: grid_min_y.clone(),
            grid_max_y: grid_max_y.clone(),
            grid_size_x: grid_size_x.clone(),
            grid_size_y: grid_size_y.clone(),
            object_cells,
            current_cell_offset: current_cell_offset.clone(),
            cell_object_count: cell_object_count.clone(),
            cell_offsets,
            cells,
            candidates,
            candidate_count: candidate_count.clone(),
            masses: masses.clone(),
            forces: forces.clone(),
            kick_center: kick_center.clone(),
            kick_radius: kick_radius.clone(),
            kick_magnitude: kick_magnitude.clone(),
        };

        let reset_grid_aabb = ResetGridAABB::new(&device, &broad_phase_buffers);
        let mut calculate_grid_aabb = CalculateGridAABB::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            particle_radius.clone(),
            &broad_phase_buffers,
            phase_state_ring_config,
        );
        let cell_offsets_dispatch_dimensions =
            CellIterationDispatchDimensions::new(&device, &broad_phase_buffers, dispatch_dimensions.clone());
        let mut assign_object_cells = AssignObjectCells::new(
            &device,
            object_count,
            object_count_buffer.clone(),
            particle_radius.clone(),
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
            particle_radius.clone(),
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
            particle_radius.clone(),
            &broad_phase_buffers,
            masses.clone(),
            forces.clone(),
            phase_state_ring_config,
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

            // Prepare the compute pass
            calculate_grid_aabb.prepare(&device, phase_state_index, &current_phase_state);
            assign_object_cells.prepare(&device, phase_state_index, &current_phase_state);
            broad_phase_grid.prepare(&device, phase_state_index, &current_phase_state);
            narrow_phase.prepare(&device, phase_state_index, &current_phase_state);
            integrator.prepare(&device, phase_state_index, &current_phase_state, &next_phase_state);

            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());

            // Reset buffers
            let timings = current_phase_state.command_timings();
            timings.measure(&mut encoder, "Clear buffers", |encoder| {
                encoder.clear_buffer(cell_object_count.buffer(), 0, None);
                encoder.clear_buffer(dispatch_dimensions.buffer(), 0, None);
                encoder.clear_buffer(current_cell_offset.buffer(), 0, None);
                encoder.clear_buffer(candidate_count.buffer(), 0, None);
                encoder.clear_buffer(forces.buffer(), 0, None);
            });

            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            compute(&reset_grid_aabb, &mut compute_pass, timings);
            compute(&calculate_grid_aabb, &mut compute_pass, timings);
            compute(&cell_offsets_dispatch_dimensions, &mut compute_pass, timings);
            compute(&assign_object_cells, &mut compute_pass, timings);
            compute(&calculate_cell_offsets, &mut compute_pass, timings);
            compute(&populate_grid_cells, &mut compute_pass, timings);
            compute(&broad_phase_grid, &mut compute_pass, timings);
            compute(&narrow_phase_dispatch_dimensions, &mut compute_pass, timings);
            compute(&narrow_phase, &mut compute_pass, timings);
            compute(&integrator, &mut compute_pass, timings);
            drop(compute_pass);

            encoder.clear_buffer(kick_magnitude.buffer(), 0, None);

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
