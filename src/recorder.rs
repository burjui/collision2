use std::sync::{Arc, Mutex};

use crossbeam::channel;
use image::{ImageBuffer, Rgba};
use wgpu::{
    BufferUsages, Device, Extent3d, Origin3d, Queue, TexelCopyBufferInfo, TexelCopyTextureInfo, Texture, TextureAspect,
    TextureDimension, TextureFormat, TextureUsages, wgt::TextureDescriptor,
};
use winit::dpi::PhysicalSize;

use crate::{
    aabb_renderer::AabbRenderer,
    camera::orthographic_camera,
    config::CONFIG,
    device_buffer::DeviceBuffer,
    phase_state::{PhaseStateRing, PhaseStateRingConfig},
    renderer::{RenderParameters, render_scene},
    shaders::common::{AABB, Camera, Color, Mass, Shape},
    shape_renderer::ShapeRenderer,
};

pub struct Recorder {
    config: RecorderConfig,
    frame_texture: Texture,
    frame_staging_buffer: DeviceBuffer<u8>,
    shape_renderer: ShapeRenderer,
    aabb_renderer: AabbRenderer,
    padded_bytes_per_row: u32,
}

impl Recorder {
    const FRAME_SIZE: PhysicalSize<u32> = PhysicalSize::new(1920, 1080);

    pub fn new(config: RecorderConfig) -> Self {
        let frame_texture = config.device.create_texture(&TextureDescriptor {
            label: Some("Frame export"),
            size: Extent3d {
                width: Self::FRAME_SIZE.width,
                height: Self::FRAME_SIZE.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let world_height = config.world_aabb.max().y - config.world_aabb.min().y;
        let camera_matrix = orthographic_camera(
            Self::FRAME_SIZE.cast(),
            world_height,
            config.render_parameters.zoom,
            config.render_parameters.offset,
        );
        let camera_buffer = DeviceBuffer::<Camera>::from_data(
            &config.device,
            &[Camera::new(camera_matrix)],
            "camera",
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        let shape_renderer = ShapeRenderer::new(
            &config.device,
            frame_texture.format(),
            camera_buffer.clone(),
            config.particle_radius.clone(),
            config.spectrum_width.clone(),
            config.colors.clone(),
            config.shapes.clone(),
            config.masses.clone(),
            config.phase_state_ring_config,
        );
        let aabb_renderer = AabbRenderer::new(
            &config.device,
            frame_texture.format(),
            camera_buffer.clone(),
            config.particle_radius.clone(),
            config.phase_state_ring_config,
        );
        let padded_bytes_per_row = (Self::FRAME_SIZE.width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let frame_staging_buffer = DeviceBuffer::<u8>::new(
            &config.device,
            padded_bytes_per_row * Self::FRAME_SIZE.height,
            "export frame staging buffer",
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        );
        if let Some(output_path) = &CONFIG.output_path {
            std::fs::create_dir_all(output_path).unwrap();
        }

        Self {
            config,
            frame_texture,
            frame_staging_buffer,
            shape_renderer,
            aabb_renderer,
            padded_bytes_per_row,
        }
    }

    pub fn record_frame(&mut self, frame_index: usize) {
        // Render to the export texture
        let texture_view = self.frame_texture.create_view(&Default::default());
        render_scene(
            &self.config.device,
            &self.config.queue,
            texture_view,
            &self.config.render_parameters,
            &mut self.shape_renderer,
            &mut self.aabb_renderer,
            &self.config.phase_state_ring,
            0..self.config.object_count,
            {
                let export_frame_texture = self.frame_texture.clone();
                let export_frame_staging_buffer = self.frame_staging_buffer.clone();
                let padded_bytes_per_row = self.padded_bytes_per_row;
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
                                rows_per_image: Some(Self::FRAME_SIZE.height),
                            },
                        },
                        Extent3d {
                            width: Self::FRAME_SIZE.width,
                            height: Self::FRAME_SIZE.height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            },
        );
        let (tx, rx) = channel::bounded(1);
        self.config.queue.on_submitted_work_done({
            let frame_staging_buffer = self.frame_staging_buffer.clone();
            let padded_bytes_per_row = self.padded_bytes_per_row;
            move || {
                assert!(padded_bytes_per_row == Self::FRAME_SIZE.width * 4);
                let frame_size_in_bytes = padded_bytes_per_row * Self::FRAME_SIZE.height;
                frame_staging_buffer.read(frame_size_in_bytes.try_into().unwrap(), move |result| {
                    let data = result.unwrap();
                    let _ = tx.send(data);
                });
            }
        });
        let data = rx.recv().unwrap();
        rayon::spawn({
            let output_path = self.config.output_path.clone();
            move || {
                let image = ImageBuffer::<Rgba<u8>, _>::from_raw(
                    Self::FRAME_SIZE.width,
                    Self::FRAME_SIZE.height,
                    data.as_slice(),
                )
                .expect("invalid image size");
                image.save(format!("{}/{:06}.png", output_path, frame_index)).unwrap();
            }
        });
    }
}

pub struct RecorderConfig {
    pub device: Device,
    pub queue: Queue,
    pub output_path: String,
    pub world_aabb: AABB,
    pub render_parameters: RenderParameters,
    pub particle_radius: DeviceBuffer<f32>,
    pub spectrum_width: DeviceBuffer<f32>,
    pub colors: DeviceBuffer<Color>,
    pub shapes: DeviceBuffer<Shape>,
    pub masses: DeviceBuffer<Mass>,
    pub phase_state_ring: Arc<Mutex<PhaseStateRing>>,
    pub phase_state_ring_config: PhaseStateRingConfig,
    pub object_count: u32,
}
