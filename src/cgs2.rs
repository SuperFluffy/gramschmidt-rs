use cblas;
use ndarray::{
    Data,
    ShapeBuilder,
};
use ndarray::prelude::*;
use std::slice;

use crate::{
    Error,
    GramSchmidt,
    Result,
    utils::{
        as_slice_with_layout,
        get_layout,
    }
};

/// A reorthogonalized Gram Schmidt factorization, also known as `CGS2` in the literature. See
/// [Giraud et al.] for a definition. It performs two successive classical Gram Schmidt procedures,
/// which has a higher performance than modified Gram Schmidt while providing a similar numerical
/// stability.
///
/// Use this struct via the [`GramSchmidt` trait].
///
/// [Giraud et al.]: https://doi.org/10.1007/s00211-005-0615-4
/// [`GramSchmidt` trait]: GramSchmidt
#[derive(Clone, Debug)]
pub struct Reorthogonalized {
    q: Array2<f64>,
    r: Array2<f64>,
    work_vector: Array1<f64>,
    memory_layout: cblas::Layout,
}

impl GramSchmidt for Reorthogonalized {
    fn from_shape<T>(shape: T) -> Result<Self>
        where T: ShapeBuilder<Dim = Ix2>,
    {
        // Unfortunately we cannot check the shape itself to see if it's
        // in ColumnMajor or RowMajor layout. So we need to first construct
        // an array and then check that.
        let shape = shape.into_shape();
        let q = Array2::zeros(shape);
        let memory_layout = match get_layout(&q) {
            Some(layout) => layout,
            None => Err(Error::NonContiguous)?,
        };
        let r = q.clone();

        // Similarly to the layout, we don't have direct access to the array dimensions via
        // `Shape`, and thus need to go via `Dim::Pattern` of the already constructed arrays.
        let work_vector = Array1::zeros(q.dim().0);

        Ok(Self {
            q,
            r,
            work_vector,
            memory_layout,
        })
    }

    fn compute<S>(&mut self, a: &ArrayBase<S, Ix2>) -> Result<()>
        where S: Data<Elem = f64>,
    {
        use cblas::Layout::*;
        use Error::*;

        assert_eq!(a.shape(), self.q.shape());

        let (n_rows, n_cols) = self.q.dim();

        let a_slice = match (self.memory_layout, as_slice_with_layout(&a)) {
            (a, Some((_, b))) if a != b => Err(IncompatibleLayouts)?,
            (_, Some((a_slice, _))) => a_slice,
            (_, None) => Err(NonContiguous)?,
        };

        // leading_dim: the number of elements in the leading dimension
        // next_elem: how many elements to jump to get to the next element in a column
        // next_col: how many elements in the array to jump to get to the next column
        let (leading_dim, next_elem, next_col) = match self.memory_layout {
            ColumnMajor => (n_rows as i32, 1, n_rows),
            RowMajor => (n_cols as i32, n_cols as i32, 1),
        };


        for i in 0..n_cols {
            self.q.column_mut(i).assign(&a.column(i));

            let len = self.q.len();
            let q_ptr = self.q.as_mut_ptr();
            let q_matrix = unsafe {
                slice::from_raw_parts(q_ptr, len)
            };

            let q_column = match self.memory_layout {
                ColumnMajor => {
                    let offset = n_rows * i;
                    unsafe {
                        slice::from_raw_parts_mut(q_ptr.offset(offset as isize), len - offset)
                    }
                },

                RowMajor => {
                    let offset = i as isize;
                    unsafe {
                        slice::from_raw_parts_mut(q_ptr.offset(offset), len - i)
                    }
                },

            };

            if i > 0 {
                let a_column = &a_slice[next_col * i..];

                // NOTE: This unwrap is save, because we have made sure at creation that r_slice is
                // contiguous.
                //
                // NOTE: Unlike a_slice above which is defined outside the loop, we are mutating r at the
                // end of the loop, which invalidates the mutable borrow. We thus have to pull the
                // slice definition into the loop.
                let r_slice = self.r.as_slice_memory_order_mut().unwrap();
                let r_column = &mut r_slice[next_col * i..];

                let work_slice = self.work_vector.as_slice_memory_order_mut().unwrap();

                unsafe {
                    // First orthogonalization
                    // =======================
                    cblas::dgemv(
                        self.memory_layout,
                        cblas::Transpose::Ordinary,
                        n_rows as i32,
                        i as i32,
                        1.0,
                        q_matrix,
                        leading_dim,
                        a_column,
                        next_elem,
                        0.0,
                        r_column,
                        next_elem
                    );

                    cblas::dgemv(
                        self.memory_layout,
                        cblas::Transpose::None,
                        n_rows as i32,
                        i as i32,
                        -1.0,
                        q_matrix,
                        leading_dim,
                        r_column,
                        next_elem,
                        1.0,
                        q_column,
                        next_elem,
                    );

                    // Second orthogonalization
                    // =======================
                    cblas::dgemv(
                        self.memory_layout,
                        cblas::Transpose::Ordinary,
                        n_rows as i32,
                        i as i32,
                        1.0,
                        q_matrix,
                        leading_dim,
                        q_column,
                        next_elem,
                        0.0,
                        work_slice,
                        1 // Always 1 from the definition of the work_slice/work_vector
                    );

                    cblas::dgemv(
                        self.memory_layout,
                        cblas::Transpose::None,
                        n_rows as i32,
                        i as i32,
                        -1.0,
                        q_matrix,
                        leading_dim,
                        work_slice,
                        1,
                        1.0,
                        q_column,
                        next_elem,
                    );

                    cblas::daxpy(
                        n_rows as i32, // n
                        1.0, // alpha
                        work_slice, // x
                        1, // Always 1 from the definition of the work_slice/work_vector
                        r_column,
                        next_elem,
                    );

                }
            };

            let norm = unsafe {
                cblas::dnrm2(n_rows as i32, q_column, next_elem)
            };

            let mut v = self.q.column_mut(i);
            v /= norm;
            self.r[(i,i)] = a.column(i).dot(&v);
        }

        Ok(())
    }

    fn q(&self) -> &Array2<f64> {
        &self.q
    }

    fn r(&self) -> &Array2<f64> {
        &self.r
    }
}

/// Convenience function that calculates a Reorthogonalized Gram Schmmidt QR factorization (see
/// [Giraud et al.] for details), returning a tuple `(Q,R)`.
///
/// If you want to repeatedly calculate QR factorizations, then prefer constructing a
/// [`Reorthogonalized`] struct and calling its [`GramSchmidt::compute`] method implemented through
/// the [`GramSchmidt`] trait.
///
/// [Giraud et al.]: https://doi.org/10.1007/s00211-005-0615-4
/// [`Reorthogonalized`]: Reorthogonalized
/// [`GramSchmidt`]: GramSchmidt
/// [`GramSchmidt::compute`]: trait.GramSchmidt.html#method.compute
pub fn cgs2<S>(a: &ArrayBase<S, Ix2>) -> Result<(Array<f64, Ix2>, Array<f64, Ix2>)>
    where S: Data<Elem=f64>
{
    let mut cgs2 = Reorthogonalized::from_matrix(a)?;
    cgs2.compute(a)?;
    Ok((cgs2.q().clone(), cgs2.r().clone()))
}

#[cfg(test)]
generate_tests!(Reorthogonalized, 1e-13);
