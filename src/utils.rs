use blas;

use ndarray::{Data,LinalgScalar};
use ndarray::prelude::*;

pub fn orthogonal<S>(a: &ArrayBase<S,Ix2>, tol: f64) -> bool
    where S: Data<Elem=f64>
{
    let b = a.dot(&a.t());
    b.all_close(&Array2::eye(b.shape()[0]), tol)
}

pub fn normalization(v: &[f64]) -> f64 {
    unsafe {
        blas::c::dnrm2(v.len() as i32, v, 1)
    }
}

pub fn project<A,S1,S2>(vector: &ArrayBase<S1,Ix1>, onto: &ArrayBase<S2,Ix1>) -> A
    where S1: Data<Elem=A>,
          S2: Data<Elem=A>,
          A: LinalgScalar
{
    vector.dot(onto) / onto.dot(onto)
}

