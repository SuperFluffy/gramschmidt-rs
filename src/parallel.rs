use ndarray::prelude::*;
use ndarray_parallel::prelude::*;

use ndarray::{Data,DataMut};
use utils::*;

pub trait ParallelModifiedGramSchmidt: Sized + Clone + Default {
    fn compute_inplace<S1,S2>(orth: &mut ArrayBase<S1, Ix2>, norm: &mut ArrayBase<S2, Ix1>)
        where S1: DataMut<Elem = Self>,
              S2: DataMut<Elem = Self>;

    fn compute_inplace_no_norm<S1>(a: &mut ArrayBase<S1, Ix2>)
        where S1: DataMut<Elem = Self>;

    fn compute_into<S1,S2,S3>(a: &ArrayBase<S1, Ix2>, orth: &mut ArrayBase<S2, Ix2>, norm: &mut ArrayBase<S3, Ix1>)
        where S1: Data<Elem = Self>,
              S2: DataMut<Elem = Self>,
              S3: DataMut<Elem = Self>,
    {
        orth.assign(&a);
        Self::compute_inplace(orth, norm);
    }

    fn compute<S>(a: &ArrayBase<S, Ix2>) -> (Array<Self, Ix2>, Array<Self, Ix1>)
        where S: Data<Elem = Self>
    {
        let mut o = a.to_owned();
        let mut n = Array1::default(a.dim().1);

        Self::compute_into(a, &mut o, &mut n);
        (o, n)
    }
}

impl ParallelModifiedGramSchmidt for f64 {
    fn compute_inplace<S1,S2>(orth: &mut ArrayBase<S1, Ix2>, norm: &mut ArrayBase<S2, Ix1>)
        where S1: DataMut<Elem = Self>,
              S2: DataMut<Elem = Self>,
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
            // orthogonalize the current row in one go and then normalize it. This however is
            // not as amenable to parallelization.
            norm[i] = normalization(v.as_slice().unwrap());
            v /= norm[i];

            todo.axis_iter_mut(Axis(0))
                .into_par_iter()
                .weight_max()
                .for_each(|mut w| {
                    // v is already normalized
                    // let projection_factor = project(&v, &w);
                    let projection_factor = v.dot(&w);
                    w.zip_mut_with(&v, |ew,ev| { *ew -= projection_factor * ev; });
            });
        }
    }

    fn compute_inplace_no_norm<S1>(orth: &mut ArrayBase<S1, Ix2>)
        where S1: DataMut<Elem = Self>,
    {
        let n_rows = orth.shape()[0];

        let mut todo = orth.view_mut();

        for _ in 0..n_rows {
            let (mut v, rest) = todo.split_at(Axis(0), 1);
            let mut v = v.row_mut(0);
            todo = rest;

            v /= normalization(v.as_slice().unwrap());

            todo.axis_iter_mut(Axis(0))
                .into_par_iter()
                .weight_max()
                .for_each(|mut w| {
                    // w is already normalized
                    // let projection_factor = project(&v, &w);
                    let projection_factor = v.dot(&w);
                    w.zip_mut_with(&v, |ew,ev| { *ew -= projection_factor * ev; });
            });
        }
    }
}
