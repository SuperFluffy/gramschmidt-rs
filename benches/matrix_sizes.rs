#![feature(test)]

#![allow(non_snake_case)]

extern crate gram_schmidt;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

extern crate test; // Built-in crate for benchmarking.

use gram_schmidt::{normalization, ModifiedGramSchmidt};
#[cfg(feature = "parallel")]
use gram_schmidt::ParallelModifiedGramSchmidt;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Normal;

/// The strategy employed in ModifiedGramSchmidt::compute_inplace is to orthogonalize the current
/// row with respect to the already-orthogonalized ones and then normalize it. We can call this
/// *late projection*, because the outstanding, not-yet orthonormalized rows are only touched after
/// the rows coming before them are completed.
///
/// On the other hand, *early projection* is the approach taken below, where we normalize the
/// current row and remove its projection from the not-yet normalized rows. This partially
/// orthogonalizes the outstanding rows with respect to each current row.
///
/// It turns out that *late projection* is about 10% faster.
fn compute_inplace_first(orth: &mut Array2<f64>, norm: &mut Array1<f64>)
{
    let n_rows = orth.shape()[0];

    let mut todo = orth.view_mut();

    for i in 0..n_rows {
        let (mut v, rest) = todo.split_at(Axis(0), 1);
        let mut v = v.row_mut(0);
        todo = rest;

        norm[i] = normalization(v.as_slice().unwrap());
        v /= norm[i];

        for mut w in todo.genrows_mut() {
            // w is already normalized
            // let projection_factor = project(&v, &w);
            let projection_factor = v.dot(&w);
            w.scaled_add(-projection_factor, &v);
        }
    }
}

macro_rules! create_bench {
    ($n:expr, $name:ident, $function:path) => {
        #[bench]
        fn $name(bench: &mut test::Bencher) {
            let n = $n;

            let dist = Normal::new(0.0, 1.0);
            let matrix = Array2::random([n,n], dist);
            let norm = Array1::zeros([n]);
            let mut matrix = test::black_box(matrix);
            let mut norm = test::black_box(norm);

            bench.iter(|| {
                $function(&mut matrix, &mut norm);
            });
        }
    }
}

macro_rules! bench_sizes {
    ($n:expr, $name_first:ident, $name_late: ident, $name_parallel: ident) => {
        create_bench!($n, $name_first, compute_inplace_first);
        create_bench!($n, $name_late, ModifiedGramSchmidt::compute_inplace);

        #[cfg(feature = "parallel")]
        create_bench!($n, $name_parallel, ParallelModifiedGramSchmidt::compute_inplace);
    }
}

bench_sizes!( 256, sequential_project_first__256, sequential_project_late__256, parallel__256);
bench_sizes!( 512, sequential_project_first__512, sequential_project_late__512, parallel__512);
bench_sizes!( 768, sequential_project_first__768, sequential_project_late__768, parallel__768);
bench_sizes!(1024, sequential_project_first_1024, sequential_project_late_1024, parallel_1024);
bench_sizes!(1536, sequential_project_first_1536, sequential_project_late_1536, parallel_1536);
bench_sizes!(2048, sequential_project_first_2048, sequential_project_late_2048, parallel_2048);
// These benches are taking too much time to have them activated by default.
// bench_sizes!(3072, sequential_project_first_3072, sequential_project_late_3072, parallel_3072);
// bench_sizes!(4096, sequential_project_first_4096, sequential_project_late_4096, parallel_4096);
// bench_sizes!(8192, sequential_project_first_8192, sequential_project_late_8192, parallel_8192);
