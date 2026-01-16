#![allow(clippy::needless_range_loop)]

use std::array::from_fn;

use crate::{
    bvh_builder::calculate_passes,
    shaders::{
        bvh::CombineNodePass,
        common::{BVH_NODE_TREE_FLAG, BvhNode},
    },
};

#[test]
fn mock_bvh() {
    const N: usize = 7;

    // expected for n = 7:
    //                               ( pass 0        ) ( pass 1  ) ( root )
    // nodes           0 1 2 3 4 5 6 [0 1] [2 3] [4 5] [6 7] [8 9] [10 11]
    // node indices    0 1 2 3 4 5 6 7     8     9     10    11    12

    let mut aabbs: [usize; N * 2] = from_fn(|i| i * usize::from(i < N)); // tree nodes set to zero
    let mut nodes = [BvhNode::new_leaf(0); N * 2];

    let mut passes = Vec::new();
    calculate_passes(N, &mut passes);
    assert_eq!(
        passes.as_slice(),
        &[
            CombineNodePass {
                src_start: 0,
                dst_start: 7,
                parent_count: 3
            },
            CombineNodePass {
                src_start: 6,
                dst_start: 10,
                parent_count: 2
            },
            CombineNodePass {
                src_start: 10,
                dst_start: 12,
                parent_count: 1
            },
        ]
    );

    for i in 0..N {
        nodes[i] = BvhNode::new_leaf(i.try_into().unwrap());
    }

    for &CombineNodePass {
        src_start,
        dst_start,
        parent_count,
    } in &passes
    {
        for i in 0..parent_count {
            let src = src_start + i * 2;
            let dst = usize::try_from(dst_start + i).unwrap();
            nodes[dst] = BvhNode::new_tree(src);
            aabbs[dst] = dst; // combine AABBs on GPU
        }
    }

    let node_count = usize::try_from(passes.last().unwrap().dst_start).unwrap();
    assert_eq!(
        &nodes[0..=node_count],
        &[
            // Leaves
            BvhNode::new_leaf(0),
            BvhNode::new_leaf(1),
            BvhNode::new_leaf(2),
            BvhNode::new_leaf(3),
            BvhNode::new_leaf(4),
            BvhNode::new_leaf(5),
            BvhNode::new_leaf(6),
            // Level 0
            BvhNode::new_tree(0),
            BvhNode::new_tree(2),
            BvhNode::new_tree(4),
            // Level 1
            BvhNode::new_tree(6),
            BvhNode::new_tree(8),
            // Root
            BvhNode::new_tree(10),
        ]
    );
    assert_eq!(aabbs.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]);
}

impl BvhNode {
    #[must_use]
    pub const fn new_leaf(index: u32) -> Self {
        Self { index }
    }

    #[must_use]
    pub const fn new_tree(left: u32) -> Self {
        Self {
            index: left | BVH_NODE_TREE_FLAG,
        }
    }
}
