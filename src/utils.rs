use ndarray::Data;
use ndarray::prelude::*;

pub(crate) fn orthogonal<S>(a: &ArrayBase<S,Ix2>, tol: f64) -> bool
    where S: Data<Elem=f64>
{
    let b = a.dot(&a.t());
    b.all_close(&Array2::eye(b.shape()[0]), tol)
}
