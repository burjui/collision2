#define_import_path collision_narrow_phase

#import common::{ Position, Mass, Velocity, CollisionCandidate, flat_invocation_index, WORKGROUP_SIZE }

@group(0) @binding(0) var<uniform> stiffness: f32;
@group(0) @binding(1) var<uniform> restitution: f32;
@group(0) @binding(2) var<uniform> particle_radius: f32;

@group(1) @binding(0) var<storage, read> positions: array<Position>;
@group(1) @binding(1) var<storage, read> velocities: array<Velocity>;
@group(1) @binding(2) var<storage, read> masses: array<Mass>;

@group(2) @binding(0) var<uniform> candidate_count: u32;
@group(2) @binding(1) var<storage, read> candidates: array<CollisionCandidate>;

@group(3) @binding(0) var<storage, read_write> forces: array<atomic<u32>>;

const STIFFNESS: f32 = 100000;
const RESTITUTION: f32 = 0.0;
const GAMMA_COEFF: f32 = (3.0 / 2.0) * (1.0 - RESTITUTION * RESTITUTION) / sqrt(5.0) * sqrt(STIFFNESS);

@compute @workgroup_size(WORKGROUP_SIZE)
fn narrow_phase(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    let candidate_index = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    if candidate_index >= candidate_count {
        return;
    }

    let candidates = candidates[candidate_index];
    let a = candidates.a;
    let b = candidates.b;
    let f = collision_repulsion_pair(
        positions[a], velocities[a].inner, masses[a].inner,
        positions[b], velocities[b].inner, masses[b].inner
    );
    cas_add_force(a, f);
    cas_add_force(b, -f);
}

fn collision_repulsion_pair(
    x1: Position,
    v1: vec2f,
    m1: f32,
    x2: Position,
    v2: vec2f,
    m2: f32,
) -> vec2f {
    let separation_vector = x1.inner - x2.inner;
    let distance = length(separation_vector);
    if (distance < 1e-10) {
        return vec2f(0.0);
    }

    let n = separation_vector / distance;
    let interaction_distance = particle_radius * 2;
    let penetration = interaction_distance - distance;
    var f_elastic = vec2f();
    f_elastic = STIFFNESS * max(0.0, penetration) * n;

    let v_rel = v1 - v2;
    let v_n = dot(v_rel, n);
    var f_damping = vec2f();
    let m_eff = effective_mass(m1, m2);
    f_damping = -GAMMA_COEFF * sqrt(m_eff) * min(0.0, v_n) * n;

    return f_elastic + f_damping;
}

fn effective_mass(m1: f32, m2: f32) -> f32 {
    if (m1 == 0.0) { return m2; }
    if (m2 == 0.0) { return m1; }
    return m1 * m2 / (m1 + m2);
}

// TODO: atomic f32

fn cas_add_force(i: u32, value: vec2f) {
    cas_add_force_component(i * 2, value.x);
    cas_add_force_component(i * 2 + 1, value.y);
}

fn cas_add_force_component(i: u32, value: f32) {
    var old = atomicLoad(&forces[i]);
    loop {
        let old_f = bitcast<f32>(old);
        let new_f = old_f + value;
        let new_value = bitcast<u32>(new_f);
        let res = atomicCompareExchangeWeak(&forces[i], old, new_value);
        if res.exchanged {
            break;
        }
        old = res.old_value;
    }
}
