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
const BATCH_SIZE: u32 = 1; // TODO make use of BATCH_SIZE

const STIFFNESS: f32 = 30000;
const RESTITUTION: f32 = 0.3;
const GAMMA_COEFF: f32 = (3.0 / 2.0) * (1.0 - RESTITUTION * RESTITUTION) / sqrt(5.0) * sqrt(STIFFNESS);

@compute @workgroup_size(WORKGROUP_SIZE)
fn narrow_phase(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    let i = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    if i >= candidate_count {
        return;
    }

    let candidates = candidates[i];
    let a = candidates.a;
    let b = candidates.b;

    let a_force_index = atomicAdd(&collision_count[a], 1);
    let b_force_index = atomicAdd(&collision_count[b], 1);
    let f = collision_repulsion_pair(aabbs[a], aabbs[b], velocities[a].inner, masses[a].inner, b);
    collision_forces[a * MAX_CANDIDATES_PER_OBJECT + a_force_index] = Force(f);
    collision_forces[b * MAX_CANDIDATES_PER_OBJECT + b_force_index] = Force(-f);
}

fn collision_repulsion_pair(aabb: AABB, other_aabb: AABB, velocity: vec2f, mass: f32, other_index: u32) -> vec2f {
    let size = aabb.max - aabb.min;
    let other_size = other_aabb.max - other_aabb.min;
    let position = (aabb.min + aabb.max) / 2;
    let other_position = (other_aabb.min + other_aabb.max) / 2;
    let separation_vector = position - other_position;
    let distance = length(separation_vector);
    let r1 = 0.5 * size.x;
    let r2 = 0.5 * other_size.x;
    let interaction_distance = r1 + r2;
    if distance >= interaction_distance {
        return vec2f();
    }

    let penetration = interaction_distance - distance;
    let n = normalize(separation_vector);
    var v_ij_n = dot(velocity - velocities[other_index].inner, n);
    if penetration <= 0 && v_ij_n > 0 {
        v_ij_n = -RESTITUTION * v_ij_n;
    }
    let m1 = mass;
    let m2 = masses[other_index].inner;
    let m_eff = m1 * m2 / (m1 + m2);
    var f_damping = vec2f();
    if v_ij_n < 0 {
        f_damping = -GAMMA_COEFF * sqrt(m_eff) * v_ij_n * n;
    }
    let f_elastic = STIFFNESS * penetration * n;
    return f_elastic + f_damping;
}