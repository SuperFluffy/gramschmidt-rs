#![feature(test)]

extern crate gram_schmidt;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate test; // Built-in crate for benchmarking.

use gram_schmidt::{normalization, ModifiedGramSchmidt};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Normal;

#[bench]
fn project_first(bench: &mut test::Bencher) {
    fn compute_inplace(orth: &mut Array2<f64>, norm: &mut Array1<f64>)
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
    let n = 1024;

    let dist = Normal::new(0.0, 1.0);
    let matrix = Array2::random([n,n], dist);
    let norm = Array1::zeros([n]);
    let mut matrix = test::black_box(matrix);
    let mut norm = test::black_box(norm);

    bench.iter(|| {
        compute_inplace(&mut matrix, &mut norm);
    });
}

#[bench]
fn project_late(bench: &mut test::Bencher) {
    // Same definition as for ModifiedGramSchmidt<f64>::compute_inplace
    fn compute_inplace(orth: &mut Array2<f64>, norm: &mut Array1<f64>)
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

    let n = 1024;

    let dist = Normal::new(0.0, 1.0);
    let matrix = Array2::random([n,n], dist);
    let norm = Array1::zeros([n]);
    let mut matrix = test::black_box(matrix);
    let mut norm = test::black_box(norm);

    bench.iter(|| {
        compute_inplace(&mut matrix, &mut norm);
    });
}
