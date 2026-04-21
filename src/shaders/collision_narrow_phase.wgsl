#define_import_path collision_narrow_phase

#import common::{
    FLAG_DRAW_OBJECT, FLAG_PHYSICAL, FLAG_DRAW_AABB, BVH_NODE_TREE_FLAG, MAX_CANDIDATES_PER_OBJECT,
    AABB, Mass, Velocity, BvhNode, CollisionCandidate, Force,
    flat_invocation_index
}

@group(0) @binding(0) var<storage, read> aabbs: array<AABB>;
@group(0) @binding(1) var<storage, read> velocities: array<Velocity>;
@group(0) @binding(2) var<storage, read> masses: array<Mass>;

@group(1) @binding(0) var<uniform> candidate_count: u32;
@group(1) @binding(1) var<storage, read> candidates: array<CollisionCandidate>;

@group(2) @binding(0) var<storage, read_write> collision_count: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> collision_forces: array<Force>;

const WORKGROUP_SIZE: u32 = 64;
const BATCH_SIZE: u32 = 1;

const STIFFNESS: f32 = 30000;
const RESTITUTION: f32 = 0.3;
const GAMMA_COEFF: f32 = (3.0 / 2.0) * (1.0 - RESTITUTION * RESTITUTION) / sqrt(5.0) * sqrt(STIFFNESS);

@compute @workgroup_size(WORKGROUP_SIZE)
fn narrow_phase(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    let fii = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    for (var batch_index: u32 = 0; batch_index < BATCH_SIZE; batch_index += 1) {
        let i = fii * BATCH_SIZE + batch_index;
        if i >= candidate_count {
            return;
        }

        let candidates = candidates[i];
        let a = candidates.a;
        let b = candidates.b;
        let f = collision_repulsion_pair(
            aabbs[a], velocities[a].inner, masses[a].inner,
            aabbs[b], velocities[b].inner, masses[b].inner
        );
        if f.x == 0 && f.y == 0 {
            return;
        }

        let a_force_index = atomicAdd(&collision_count[a], 1);
        let b_force_index = atomicAdd(&collision_count[b], 1);
        collision_forces[a * MAX_CANDIDATES_PER_OBJECT + a_force_index] = Force(f);
        collision_forces[b * MAX_CANDIDATES_PER_OBJECT + b_force_index] = Force(-f);
    }

}

fn collision_repulsion_pair(
    aabb1: AABB,
    v1: vec2f,
    m1: f32,
    aabb2: AABB,
    v2: vec2f,
    m2: f32,
) -> vec2f {
    let size1 = aabb1.max - aabb1.min;
    let size2 = aabb2.max - aabb2.min;
    let x1 = (aabb1.min + aabb1.max) * 0.5;
    let x2 = (aabb2.min + aabb2.max) * 0.5;
    let separation_vector = x1 - x2;
    let distance = length(separation_vector);
    let r1 = 0.5 * size1.x;
    let r2 = 0.5 * size2.x;
    let interaction_distance = r1 + r2;

    if distance >= interaction_distance {
        return vec2f();
    }

    if distance == 0.0 {
        return vec2f();
    }

    let n = separation_vector / distance;
    let penetration = interaction_distance - distance;
    let f_elastic = STIFFNESS * penetration * n;

    let v_rel = v1 - v2;
    let v_n = dot(v_rel, n);
    var f_damping = vec2f();
    if v_n < 0.0 {
        let m_eff = m1 * m2 / (m1 + m2);
        f_damping = -GAMMA_COEFF * sqrt(m_eff) * v_n * n;
    }

    return f_elastic + f_damping;
}
