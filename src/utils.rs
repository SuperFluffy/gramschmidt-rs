use ndarray::Data;
use ndarray::prelude::*;

#[cfg(test)]
pub(crate) fn orthogonal<S>(a: &ArrayBase<S,Ix2>, tol: f64) -> bool
    where S: Data<Elem=f64>
{
    let b = a.dot(&a.t());
    b.all_close(&Array2::eye(b.shape()[0]), tol)
}

/// Returns slice and layout underlying an array `a`.
pub(crate) fn get_layout<S, T, D>(a: &ArrayBase<S, D>) -> Option<cblas::Layout>
    where S: Data<Elem=T>,
          D: Dimension
{
    if let Some(_) = a.as_slice() {
        Some(cblas::Layout::RowMajor)
    } else if let Some(_) = a.as_slice_memory_order() {
        Some(cblas::Layout::ColumnMajor)
    } else {
        None
    }
}

/// Returns slice and layout underlying an array `a`.
pub(crate) fn as_slice_with_layout<S, T, D>(a: &ArrayBase<S, D>) -> Option<(&[T], cblas::Layout)>
    where S: Data<Elem=T>,
          D: Dimension
{
    if let Some(a_slice) = a.as_slice() {
        Some((a_slice, cblas::Layout::RowMajor))
    } else if let Some(a_slice) = a.as_slice_memory_order() {
        Some((a_slice, cblas::Layout::ColumnMajor))
    } else {
        None
    }
}
