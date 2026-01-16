#import common::{ AABB, BvhNode, invocation_index, BVH_NODE_TREE_FLAG }

// TODO: mini-BVHs on a grid

struct CombineNodePass {
    src_start: u32,
    dst_start: u32,
    parent_count: u32,
}

var<push_constant> params: CombineNodePass;
@group(0) @binding(0) var<storage, read_write> aabbs: array<AABB>;
@group(0) @binding(1) var<storage, read_write> nodes: array<BvhNode>;

const WORKGROUP_SIZE: u32 = 64;

@compute @workgroup_size(WORKGROUP_SIZE)
fn combine_nodes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = invocation_index(gid, WORKGROUP_SIZE);
    if (index >= params.parent_count) {
        return;
    }

    let src = params.src_start + index * 2;
    let dst = params.dst_start + index;
    nodes[dst] = BvhNode(src | BVH_NODE_TREE_FLAG);
    let left_aabb = aabbs[src];
    let right_aabb = aabbs[src + 1];
    let aabb_min = min(left_aabb.min, right_aabb.min);
    let aabb_max = max(left_aabb.max, right_aabb.max);
    aabbs[dst] = AABB(aabb_min, aabb_max);
}
