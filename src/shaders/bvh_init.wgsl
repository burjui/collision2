#import common::{AABB, BvhNode, invocation_index, BVH_NODE_KIND_LEAF, BVH_NODE_KIND_TREE}

@group(0) @binding(0) var<storage, read> aabbs: array<AABB>;
@group(0) @binding(1) var<storage, read> layers: array<vec2u>;
@group(0) @binding(2) var<storage, read_write> nodes: array<BvhNode>;

const WORKGROUP_SIZE: u32 = 64;

@compute @workgroup_size(WORKGROUP_SIZE)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // let i = invocation_index(gid, WORKGROUP_SIZE);
    // if i >= arrayLength(&nodes) {
    //     return;
    // }

    // let aabb_count = arrayLength(&aabbs);
    // var node: BvhNode;
    // if i < aabb_count {
    //     node = BvhNode(
    //         BVH_NODE_KIND_LEAF,
    //         aabbs[i],
    //         0,
    //         0,
    //     );
    // } else {
    //     let layer_count = arrayLength(&layers);
    //     var layer_index: u32 = 0;
    //     while layer_index < layer_count {
    //         let layer = layers[layer_index];
    //         if i < layer.x + layer.y {
    //             break;
    //         }
    //         layer_index += 1;
    //     }
    //     let
    //     node = BvhNode(
    //         BVH_NODE_KIND_TREE,
    //         AABB(vec2f(), vec2f()),
    //         i - aabb_count,
    //         0,
    //     );
    // }

    // var start: u32 = 0;
    // var end = arrayLength(&aabbs);
    // while end > start {
    // }
}
