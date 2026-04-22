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
        .workspace_root("src/shaders")
        .add_entry_point("src/shaders/common.wgsl")
        .add_entry_point("src/shaders/render_shape.wgsl")
        .add_entry_point("src/shaders/render_aabb.wgsl")
        .add_entry_point("src/shaders/build_bvh.wgsl")
        .add_entry_point("src/shaders/collision_broad_phase.wgsl")
        .add_entry_point("src/shaders/collision_narrow_phase_dispatch_dimensions.wgsl")
        .add_entry_point("src/shaders/collision_forces_reset.wgsl")
        .add_entry_point("src/shaders/collision_narrow_phase.wgsl")
        .add_entry_point("src/shaders/integrate.wgsl")
        .ir_capabilities(WgslShaderIrCapabilities::PUSH_CONSTANT)
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .type_map(NalgebraWgslTypeMap)
        .output("src/shaders.rs")
        .build()?
        .generate()?;
    Ok(())
}
