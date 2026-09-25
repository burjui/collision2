use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

use nalgebra::Vector2;
use wgpu::{
    CommandEncoder, CommandEncoderDescriptor, Device, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    TextureView,
};

use crate::{
    aabb_renderer::AabbRenderer, command_timings::CommandTimings, config::CONFIG, phase_state::PhaseStateRing,
    shape_renderer::ShapeRenderer,
};

#[derive(Copy, Clone)]
pub struct RenderParameters {
    pub enabled: bool,
    pub draw_aabbs: bool,
    pub zoom: f32,
    pub offset: Vector2<f32>,
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

pub fn render_scene(
    device: &Device,
    queue: &Queue,
    surface_texture_view: TextureView,
    render_parameters: &RenderParameters,
    shape_renderer: &mut ShapeRenderer,
    aabb_renderer: &mut AabbRenderer,
    phase_state_ring: &Arc<Mutex<PhaseStateRing>>,
    instances: Range<u32>,
    before_submit: impl FnOnce(&mut CommandEncoder),
    label: &'static str,
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

    let mut timings = CommandTimings::new(device, 2);

    if render_parameters.enabled {
        shape_renderer.prepare(current_frame_index, device, &current_frame);
        timings.measure_render(&mut render_pass, "shapes", |render_pass| {
            shape_renderer.render(render_pass, instances.clone())
        });
    }
    if render_parameters.draw_aabbs {
        aabb_renderer.prepare(current_frame_index, device, &current_frame);
        timings.measure_render(&mut render_pass, "aabbs", |render_pass| aabb_renderer.render(render_pass, instances));
    }

    drop(render_pass);

    let timings_reader = timings.resolve(&mut encoder, queue.get_timestamp_period());
    before_submit(&mut encoder);
    queue.submit([encoder.finish()]);
    if CONFIG.printouts {
        queue.on_submitted_work_done(move || {
            timings_reader.read(move |timings| {
                for (stage, duration) in timings {
                    println!("{label}[{stage}]: {duration:?}");
                }
            });
        });
    }
}
