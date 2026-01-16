#![allow(clippy::needless_range_loop)]

use std::array::from_fn;

use crate::{bvh::calculate_passes, shaders::bvh::CombineNodePass};

#[test]
fn mock_bvh() {
    const N: usize = 7;

    // expected for n = 7:
    //                               ( pass 0        ) ( pass 1  ) ( root )
    // nodes           0 1 2 3 4 5 6 [0 1] [2 3] [4 5] [6 7] [8 9] [10 11]
    // node indices    0 1 2 3 4 5 6 7     8     9     10    11    12

    let mut aabbs: [usize; N * 2] = from_fn(|i| i * usize::from(i < N)); // tree nodes set to zero
    let mut nodes = [Node::Object(0); N * 2];

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
        nodes[i] = Node::Object(i);
    }

    for &CombineNodePass {
        src_start,
        dst_start,
        parent_count,
    } in &passes
    {
        for i in 0..parent_count {
            let src = usize::try_from(src_start + i * 2).unwrap();
            let dst = usize::try_from(dst_start + i).unwrap();
            nodes[dst] = Node::Tree((src, src + 1));
            aabbs[dst] = dst; // combine AABBs on GPU
        }
    }

    let node_count = usize::try_from(passes.last().unwrap().dst_start).unwrap();
    assert_eq!(
        &nodes[0..=node_count],
        &[
            // Leaves
            Node::Object(0),
            Node::Object(1),
            Node::Object(2),
            Node::Object(3),
            Node::Object(4),
            Node::Object(5),
            Node::Object(6),
            // Level 0
            Node::Tree((0, 1)),
            Node::Tree((2, 3)),
            Node::Tree((4, 5)),
            // Level 1
            Node::Tree((6, 7)),
            Node::Tree((8, 9)),
            // Root
            Node::Tree((10, 11)),
        ]
    );
    assert_eq!(aabbs.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]);
}

// TODO: remove this and use BvhNode with the high bit trick
#[derive(Debug, Clone, Copy, PartialEq)]
enum Node {
    Object(usize),
    Tree((usize, usize)),
}
