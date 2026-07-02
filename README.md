An experiment in 2D GPU-based particle system simulation powered by [`wgpu`](https://github.com/gfx-rs/wgpu) and the awesome [`wgsl-bindgen`](https://github.com/Swoorup/wgsl-bindgen).

_Everything_ is done on the GPU:
- SDF-based rendering: 6 million particles at 60 FPS on Radeon RX 560.
- Grid-based collision broad phase, currently implemented using WGSL atomics (more efficient prefix sum implementation based on subrgroup operations is in the works).
- Efficient collision narrow phase employing bitcasts and CAS loop to store f32 forces.
- Symplectic Euler integrator.
- "Black holes"! Only gravitational attraction and frame dragging are implemented, no Minkowsky space-time or anything like that.

Note:
- Not published on crates.io since it's not a tool
- Only tested on Manjaro Linux with RADV on RX 560.
- The code barely has any comments, because it requires some proficiency in GPU programming and the lion's share of it is self-explanatory if you have it. And I'm lazy.