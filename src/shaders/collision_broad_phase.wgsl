#import common::{
    FLAG_PHYSICAL, BVH_NODE_TREE_FLAG, CANDIDATES_PER_OBJECT,
    AABB, Flags, BvhNode, CollisionCandidate,
    flat_invocation_index
}

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<uniform> max_candidates: u32;
@group(0) @binding(2) var<storage, read_write> candidates: array<CollisionCandidate>;
@group(0) @binding(3) var<storage, read_write> candidate_count: atomic<u32>;
@group(0) @binding(4) var<storage, read> nodes: array<BvhNode>;

@group(1) @binding(1) var<storage, read> aabbs: array<AABB>;
@group(1) @binding(2) var<storage, read> flags: array<Flags>;

const WORKGROUP_SIZE: u32 = 64;
const BATCH_SIZE: u32 = 1; // TODO make use of BATCH_SIZE
const MAX_WG_CANDIDATES: u32 = WORKGROUP_SIZE * CANDIDATES_PER_OBJECT;

var<workgroup> wg_candidate_count: atomic<u32>;
var<workgroup> wg_candidates: array<CollisionCandidate, MAX_WG_CANDIDATES>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn broad_phase(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    let i = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    if i >= object_count {
        return;
    }

    if local_invocation_index == 0 {
        atomicStore(&wg_candidate_count, 0);
    }

    workgroupBarrier();

    const MAX_STACK_DEPTH: u32 = 64; // 2 * max tree depth

    var stack: array<u32, MAX_STACK_DEPTH>;
    var sp: u32 = 0;

    stack[sp] = arrayLength(&nodes) - 1; // root
    sp++;

    let aabb = aabbs[i];

    while sp > 0 {
        let node_index = stack[sp - 1];
        sp--;

        let other_aabb = aabbs[node_index];
        if !aabb_overlaps(aabb, other_aabb) {
            continue;
        }

        let other_index = nodes[node_index].index;
        if (other_index & BVH_NODE_TREE_FLAG) != 0 {
            if sp >= MAX_STACK_DEPTH - 2 {
                break;
            }

            let child = other_index & ~BVH_NODE_TREE_FLAG;
            stack[sp] = child;
            stack[sp + 1] = child + 1;
            sp += 2;
        } else if other_index != i && (flags[other_index].inner & FLAG_PHYSICAL) != 0 {
            let candidates_index = atomicAdd(&wg_candidate_count, 1);
            if candidates_index >= MAX_WG_CANDIDATES {
                wg_candidates[candidates_index] = CollisionCandidate(i, other_index);
            }
        }
    }

    workgroupBarrier();

    if local_invocation_index == 0 {
        let base = atomicAdd(&candidate_count, wg_candidate_count);
        for (var j = 0u; j < wg_candidate_count; j++) {
            candidates[base + j] = wg_candidates[j];
        }
    }
}

fn aabb_overlaps(a: AABB, b: AABB) -> bool {
    return a.min.x < b.max.x &&
           a.max.x > b.min.x &&
           a.min.y < b.max.y &&
           a.max.y > b.min.y;
}
