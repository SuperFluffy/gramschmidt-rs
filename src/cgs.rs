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
    },
};

/// A classical Gram Schmidt factorization. See the [Gram Schmidt Wikipedia entry] for more information.
///
/// Use this struct via the [`GramSchmidt` trait].
///
/// [Gram Schmidt Wikipedia entry]: https://en.wikipedia.org/wiki/Gram-Schmidt_process
/// [`GramSchmidt` trait]: GramSchmidt
#[derive(Clone, Debug)]
pub struct Classical {
    q: Array2<f64>,
    r: Array2<f64>,
    memory_layout: cblas::Layout,
}

impl GramSchmidt for Classical {
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
        Ok(Self {
            q,
            r,
            memory_layout,
        })
    }

    fn compute<S>(&mut self, a: &ArrayBase<S, Ix2>) -> Result<()>
        where S: Data<Elem = f64>
    {
        use cblas::Layout::*;
        use Error::*;

        assert_eq!(a.shape(), self.q.shape());

        let (n_rows, n_cols) = self.q.dim();

        let a_slice = match (self.memory_layout, as_slice_with_layout(a)) {
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

            // The unsafe blocks below are because we need several overlapping slices into the
            // q matrix. The mutable `q_column` is the i-th orthogonal vector which is currently
            // being sought. The immutable `q` represents the already found vectors (vectors 0
            // to i-1). Appropriately choosing the offset, lda, and increment makes sure that
            // blas does not access the same memory location.
            //
            // This is only necessary for row major formats. In column major formats there
            // won't be overlaps.

            let q_len = self.q.len();
            let q_ptr = self.q.as_mut_ptr();
            let q_matrix = unsafe {
                slice::from_raw_parts(q_ptr, q_len)
            };

            let q_column = match self.memory_layout {
                ColumnMajor => {
                    let offset = n_rows * i;
                    unsafe {
                        slice::from_raw_parts_mut(q_ptr.offset(offset as isize), q_len - offset)
                    }
                },

                RowMajor => {
                    let offset = i as isize;
                    unsafe {
                        slice::from_raw_parts_mut(q_ptr.offset(offset), q_len - i)
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

                // Calculate the product R_(i) = Q^T·A_(i), where A_(i) is the i-th column of the matrix A,
                // and R_(i) is the i-th column of matrix R.
                unsafe {
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
                        next_elem,
                    );

                    // Calculate Q_(i) = A_(i) - Q · R_(i) = A_(i) - Q · (Q^T · A_(i)), where
                    // Q · (Q^T ·A_(i)) is the projection of the i-th column of A onto the already
                    // orthonormalized basis vectors Q_{0..i}.
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

#[cfg(test)]
generate_tests!(Classical, 1e-12);
