#![feature(test)]

extern crate gram_schmidt;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate test; // Built-in crate for benchmarking.

use gram_schmidt::parallel::ParallelModifiedGramSchmidt;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Normal;

#[bench]
fn parallel(bench: &mut test::Bencher) {
    let n = 1024;

    let dist = Normal::new(0.0, 1.0);
    let matrix = Array2::random([n,n], dist);
    let norm = Array1::zeros([n]);
    let mut matrix = test::black_box(matrix);
    let mut norm = test::black_box(norm);

    bench.iter(|| {
        ParallelModifiedGramSchmidt::compute_inplace(&mut matrix, &mut norm);
    });
}
