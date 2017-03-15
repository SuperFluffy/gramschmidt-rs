#![feature(test)]

#![allow(non_snake_case)]

extern crate gram_schmidt;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate rayon;
extern crate test; // Built-in crate for benchmarking.

use gram_schmidt::{normalization, ModifiedGramSchmidt};
use gram_schmidt::parallel::ParallelModifiedGramSchmidt;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Normal;

fn compute_inplace_first(orth: &mut Array2<f64>, norm: &mut Array1<f64>)
{
    let n_rows = orth.shape()[0];

    let mut todo = orth.view_mut();

    for i in 0..n_rows {
        let (mut v, rest) = todo.split_at(Axis(0), 1);
        let mut v = v.row_mut(0);
        todo = rest;

        // Normalize the current row and remove its projection from the remaining rows. This
        // partially orthogonalizes the outstanding rows with respect to each current row.
        //
        // Another strategy would have been to use the already orthogonalized rows to
        // orthogonalize the current row in one go and then normalize it.
        norm[i] = normalization(v.as_slice().unwrap());
        v /= norm[i];

        for mut w in todo.inner_iter_mut() {
            // w is already normalized
            // let projection_factor = project(&v, &w);
            let projection_factor = v.dot(&w);
            w.zip_mut_with(&v, |ew,ev| { *ew -= projection_factor * ev; });
        }
    }
}

// Same definition as for ModifiedGramSchmidt<f64>::compute_inplace
fn compute_inplace_late(orth: &mut Array2<f64>, norm: &mut Array1<f64>)
{
    let n_rows = orth.shape()[0];

    for i in 0..n_rows {
        let (done, mut todo) = orth.view_mut().split_at(Axis(0), i);

        let mut v = todo.row_mut(0);

        for w in done.inner_iter() {
            // w is already normalized
            // let projection_factor = project(&v, &w);
            let projection_factor = v.dot(&w);
            v.zip_mut_with(&w, |ev,ew| { *ev -= projection_factor * ew; });
        }

        norm[i] = normalization(v.as_slice().unwrap());
        v /= norm[i];
    }
}

macro_rules! bench_sizes {
    ($n:expr, $name_first:ident, $name_late: ident, $name_parallel: ident) => {
    #[bench]
    fn $name_first(bench: &mut test::Bencher) {
        let n = $n;

        let dist = Normal::new(0.0, 1.0);
        let matrix = Array2::random([n,n], dist);
        let norm = Array1::zeros([n]);
        let mut matrix = test::black_box(matrix);
        let mut norm = test::black_box(norm);

        bench.iter(|| {
            compute_inplace_first(&mut matrix, &mut norm);
        });
    }

    #[bench]
    fn $name_late(bench: &mut test::Bencher) {

        let n = $n;

        let dist = Normal::new(0.0, 1.0);
        let matrix = Array2::random([n,n], dist);
        let norm = Array1::zeros([n]);
        let mut matrix = test::black_box(matrix);
        let mut norm = test::black_box(norm);

        bench.iter(|| {
            compute_inplace_late(&mut matrix, &mut norm);
        });
    }

    #[bench]
    fn $name_parallel(bench: &mut test::Bencher) {
        rayon::initialize(rayon::Configuration::new().set_num_threads(2));

        let n = $n;

        let dist = Normal::new(0.0, 1.0);
        let matrix = Array2::random([n,n], dist);
        let norm = Array1::zeros([n]);
        let mut matrix = test::black_box(matrix);
        let mut norm = test::black_box(norm);

        bench.iter(|| {
            ParallelModifiedGramSchmidt::compute_inplace(&mut matrix, &mut norm);
        });
    }
}}

bench_sizes!( 512, sequential_project_first__512, sequential_project_late__512, parallel__512);
bench_sizes!( 768, sequential_project_first__768, sequential_project_late__768, parallel__768);
bench_sizes!(1024, sequential_project_first_1024, sequential_project_late_1024, parallel_1024);
bench_sizes!(1536, sequential_project_first_1536, sequential_project_late_1536, parallel_1536);
// bench_sizes!(2048, sequential_project_first_2048, sequential_project_late_2048, parallel_2048);
// bench_sizes!(3072, sequential_project_first_3072, sequential_project_late_3072, parallel_3072);
// bench_sizes!(4096, sequential_project_first_4096, sequential_project_late_4096, parallel_4096);
// bench_sizes!(8192, sequential_project_first_8192, sequential_project_late_8192, parallel_8192);
