Self-nominating [collision2](https://github.com/burjui/collision2).

It's an experiment in 2D GPU-based particle system simulation powered by [`wgpu`](https://github.com/gfx-rs/wgpu) and the awesome [`wgsl-bindgen`](https://github.com/Swoorup/wgsl-bindgen).

_Everything_ is done on the GPU: no CPU<->GPU data transfer after the data is uploaded at startup.
- SDF-based rendering: 6 million particles at 60 FPS on Radeon RX 560.
  6M is only rendering though, physics performance is nowhere near that.
- Grid-based collision broad phase, currently implemented using WGSL atomics (working on a more efficient prefix sum implementation utilising subgroup operations).
- Efficient collision narrow phase employing bitcasts and CAS loop to store f32 forces.
- Symplectic Euler integrator.
- "Black holes"! Only gravitational attraction and frame dragging (Lense-Thirring) are implemented, no Minkowsky space-time or anything like that. Check out `integrator.rs` and the corresponding `integrate.wgsl` for details.
- Zoom with mouse wheel.

Notes
- Grid-based broad phase assumes particles of similar sizes. Performance degrades quickly if they differ significantly. There was a BVH-based broad phase, but I removed it for now. Look up the repo history if you want to check it out.
- The code barely has any comments, because it requires some proficiency in GPU programming and the lion's share of it is self-explanatory if you have it. And I'm lazy.
- Simulation step is the DT constant in `main.rs`
- Indirect dispatch is utilised where appropriate to avoid CPU<->GPU data transfer.
- Not published on crates.io since it's not a tool.
- Only tested on Manjaro Linux with RADV Vulkan driver on RX 560.
- No config file for now. If you want to change the simulation parameters, feel free to explore the code. Sorry.
- Definitely has bugs. If I had a dollar for each one I've found over the course of development, I could easily fund my beer-brewing hobby 😆

Any feedback is welcome, especially from GPU folks.