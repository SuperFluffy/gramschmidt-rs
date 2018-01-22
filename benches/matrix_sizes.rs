#![feature(test)]

#![allow(non_snake_case)]

extern crate gram_schmidt;
extern crate ndarray;
extern crate openblas_src;

extern crate test; // Built-in crate for benchmarking.

use gram_schmidt::{
    ClassicalGramSchmidt,
    ModifiedGramSchmidt,
    ReorthogonalizedGramSchmidt,
};

use ndarray::prelude::*;

// The strategy employed in ModifiedGramSchmidt::compute_inplace is to orthogonalize the current
// row with respect to the already-orthogonalized ones and then normalize it. We can call this
// *late projection*, because the outstanding, not-yet orthonormalized rows are only touched after
// the rows coming before them are completed.
//
// On the other hand, *early projection* is the approach taken below, where we normalize the
// current row and remove its projection from the not-yet normalized rows. This partially
// orthogonalizes the outstanding rows with respect to each current row.
//
// It turns out that *late projection* is about 10% faster.
//
// fn compute_inplace_first(orth: &mut Array2<f64>, norm: &mut Array1<f64>)
// {
//     let n_rows = orth.shape()[0];

//     let mut todo = orth.view_mut();

//     for i in 0..n_rows {
//         let (mut v, rest) = todo.split_at(Axis(0), 1);
//         let mut v = v.row_mut(0);
//         todo = rest;

//         norm[i] = normalization(v.as_slice().unwrap());
//         v /= norm[i];

//         for mut w in todo.genrows_mut() {
//             // w is already normalized
//             // let projection_factor = project(&v, &w);
//             let projection_factor = v.dot(&w);
//             w.scaled_add(-projection_factor, &v);
//         }
//     }
// }

macro_rules! create_bench {
    (c $n:expr, $name:ident, $method_constructor:path) => {
        #[bench]
        fn $name(bench: &mut test::Bencher) {
            let n = $n;

            let matrix = Array2::eye(n);
            let mut method = $method_constructor(&matrix);
            let method = test::black_box(&mut method);

            bench.iter(|| {
                method.compute(&matrix);
            });
        }
    };

    (f $n:expr, $name:ident, $method_constructor:path) => {
        #[bench]
        fn $name(bench: &mut test::Bencher) {
            let n = $n;

            let matrix = Array2::eye(n).reversed_axes();
            let mut method = $method_constructor(&matrix);
            let method = test::black_box(&mut method);

            bench.iter(|| {
                method.compute(&matrix);
            });
        }
    };
}

macro_rules! bench_sizes {
    (c $n:expr, $name_cgs:ident, $name_mgs: ident, $name_cgs2: ident) => {
        create_bench!(c $n, $name_cgs, ClassicalGramSchmidt::from_matrix);
        create_bench!(c $n, $name_cgs2, ReorthogonalizedGramSchmidt::from_matrix);
        create_bench!(c $n, $name_mgs, ModifiedGramSchmidt::from_matrix);
    };

    (f $n:expr, $name_cgs:ident, $name_mgs: ident, $name_cgs2: ident) => {
        create_bench!(f $n, $name_cgs, ClassicalGramSchmidt::from_matrix);
        create_bench!(f $n, $name_cgs2, ReorthogonalizedGramSchmidt::from_matrix);
        create_bench!(f $n, $name_mgs, ModifiedGramSchmidt::from_matrix);
    };
}

bench_sizes!(c  256, c_cgs__256, c_mgs__256, c_cgs2__256);
bench_sizes!(c  512, c_cgs__512, c_mgs__512, c_cgs2__512);
bench_sizes!(c  768, c_cgs__768, c_mgs__768, c_cgs2__768);
bench_sizes!(c 1024, c_cgs_1024, c_mgs_1024, c_cgs2_1024);
// bench_sizes!(c 1536, c_cgs_1536, c_mgs_1536, c_cgs2_1536);
// bench_sizes!(c 2048, c_cgs_2048, c_mgs_2048, c_cgs2_2048);

bench_sizes!(f  256, f_cgs__256, f_mgs__256, f_cgs2__256);
bench_sizes!(f  512, f_cgs__512, f_mgs__512, f_cgs2__512);
bench_sizes!(f  768, f_cgs__768, f_mgs__768, f_cgs2__768);
bench_sizes!(f 1024, f_cgs_1024, f_mgs_1024, f_cgs2_1024);
// bench_sizes!(f 1536, f_cgs_1536, f_mgs_1536, f_cgs2_1536);
// bench_sizes!(f 2048, f_cgs_2048, f_mgs_2048, f_cgs2_2048);
