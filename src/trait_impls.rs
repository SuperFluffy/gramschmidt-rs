use cblas;
use ndarray::{Data,DataMut};
use ndarray::prelude::*;

use traits::*;
use utils::*;

impl ModifiedGramSchmidt for f64 {
    fn compute_inplace<S1,S2>(orth: &mut ArrayBase<S1, Ix2>, norm: &mut ArrayBase<S2, Ix1>)
        where S1: DataMut<Elem = Self>,
              S2: DataMut<Elem = Self>,
    {
        let n_rows = orth.shape()[0];

        // Orthonormalize the current row with respect to all already orthonormalized rows.
        //
        // Another strategy would have been to normalize the current row, and then remove it
        // from all not-yet-orthonormalized rows. However, benchmarking reveals that the first
        // strategy is about 10% faster.
        for i in 0..n_rows {
            let (done, mut todo) = orth.view_mut().split_at(Axis(0), i);

            let mut v = todo.row_mut(0);

            for w in done.genrows() {
                // w is already normalized
                // let projection_factor = project(&v, &w);
                let projection_factor = v.dot(&w);
                v.scaled_add(-projection_factor, &w);
            }

            norm[i] = normalization(v.as_slice().unwrap());
            v /= norm[i];
        }
    }

    fn compute_inplace_no_norm<S1>(orth: &mut ArrayBase<S1, Ix2>)
        where S1: DataMut<Elem = Self>,
    {
        let n_rows = orth.shape()[0];

        for i in 0..n_rows {
            let (done, mut todo) = orth.view_mut().split_at(Axis(0), i);

            let mut v = todo.row_mut(0);

            for w in done.genrows() {
                // w is already normalized
                // let projection_factor = project(&v, &w);
                let projection_factor = v.dot(&w);
                v.scaled_add(-projection_factor, &w);
            }

            v /= normalization(v.as_slice().unwrap());
        }
    }
}
