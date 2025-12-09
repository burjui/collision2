use wgsl_bindgen::{NalgebraWgslTypeMap, WgslBindgenOptionBuilder, WgslTypeSerializeStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    WgslBindgenOptionBuilder::default()
        .workspace_root("shaders")
        .add_entry_point("shaders/circle.wgsl")
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .type_map(NalgebraWgslTypeMap)
        .output("src/shaders.rs")
        .build()?
        .generate()?;
    Ok(())
}
