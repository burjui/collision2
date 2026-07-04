use std::process::exit;

use wgsl_bindgen::{
    NalgebraWgslTypeMap, WgslBindgenOptionBuilder, WgslShaderIrCapabilities, WgslTypeSerializeStrategy,
};

fn main() {
    if let Err(error) = generate_shader_bindings() {
        println!("Failed to generate shader bindings\n{error}");
        exit(1);
    }
}

fn generate_shader_bindings() -> Result<(), Box<dyn std::error::Error>> {
    WgslBindgenOptionBuilder::default()
        .ir_capabilities(WgslShaderIrCapabilities::IMMEDIATES | WgslShaderIrCapabilities::SUBGROUP)
        .workspace_root("src/shaders")
        .add_entry_point("src/shaders/common.wgsl")
        .add_entry_point("src/shaders/render_shape.wgsl")
        .add_entry_point("src/shaders/render_aabb.wgsl")
        .add_entry_point("src/shaders/reset_grid_aabb.wgsl")
        .add_entry_point("src/shaders/calculate_grid_aabb.wgsl")
        .add_entry_point("src/shaders/assign_object_cells.wgsl")
        .add_entry_point("src/shaders/calculate_cell_offsets_dispatch_dimensions.wgsl")
        .add_entry_point("src/shaders/calculate_cell_offsets.wgsl")
        .add_entry_point("src/shaders/populate_grid_cells.wgsl")
        .add_entry_point("src/shaders/collision_broad_phase_grid.wgsl")
        .add_entry_point("src/shaders/collision_narrow_phase_dispatch_dimensions.wgsl")
        .add_entry_point("src/shaders/collision_forces_reset.wgsl")
        .add_entry_point("src/shaders/collision_narrow_phase.wgsl")
        .add_entry_point("src/shaders/integrate.wgsl")
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .type_map(NalgebraWgslTypeMap)
        .output("src/shaders.rs")
        .build()?
        .generate()?;
    Ok(())
}
