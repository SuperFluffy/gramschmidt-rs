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
    utils::get_layout,
};

/// A modified Gram Schmidt factorization, which has a better numerical stability compared to
/// the classical Gram Schmidt procedure. See its [Wikipedia entry] for more information.
///
/// Use this struct via the [`GramSchmidt` trait].
///
/// [Wikipedia entry]: https://en.wikipedia.org/wiki/Gram-Schmidt_process#Numerical_stabilty
/// [`GramSchmidt` trait]: GramSchmidt
#[derive(Clone, Debug)]
pub struct Modified {
    q: Array2<f64>,
    r: Array2<f64>,
    memory_layout: cblas::Layout,
}

impl GramSchmidt for Modified {
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
        where S: Data<Elem = f64>,
    {
        let (n_rows, n_cols) = a.dim();

        // Orthonormalize the current row with respect to all already orthonormalized rows.
        //
        // Another strategy would have been to normalize the current row, and then remove it
        // from all not-yet-orthonormalized rows. However, benchmarking reveals that the first
        // strategy is about 10% faster.
        for i in 0..n_cols {
            {
                let (q_done, mut q_todo) = self.q.view_mut().split_at(Axis(1), i);
                let mut q_todo_column = q_todo.column_mut(0);
                q_todo_column.assign(&a.column(i));

                for (j, q_done_column) in q_done.gencolumns().into_iter().enumerate() {
                    let projection_factor = q_done_column.dot(&q_todo_column);
                    self.r[(j, i)] = projection_factor;
                    q_todo_column.scaled_add(-projection_factor, &q_done_column);
                }
            }

            let norm = {
                let len = self.q.len();
                let q_ptr = self.q.as_mut_ptr();
                unsafe {
                    let (q_column, q_inc) = match self.memory_layout {
                        cblas::Layout::RowMajor => {
                            let offset = i as isize;
                            let q_column = slice::from_raw_parts_mut(q_ptr.offset(offset), len - i);
                            (q_column, n_cols as i32)
                        },

                        cblas::Layout::ColumnMajor => {
                            let offset = n_rows * i;
                            let q_column = slice::from_raw_parts_mut(q_ptr.offset(offset as isize), len - offset);
                            (q_column, 1)
                        },
                    };
                    cblas::dnrm2(n_rows as i32, q_column, q_inc)
                }
            };

            self.r[(i,i)] = norm;
            let mut q_column = self.q.column_mut(i);
            q_column /= norm;
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

/// Convenience function that calculates a [Modified Gram Schmidt] QR factorization, returning a
/// tuple `(Q,R)`.
///
/// If you want to repeatedly calculate QR factorizations, then prefer constructing a
/// [`Modified`] struct and calling its [`GramSchmidt::compute`] method implemented through
/// the [`GramSchmidt`] trait.
///
/// [Modified Gram Schmidt]: https://en.wikipedia.org/wiki/Gram-Schmidt_process#Numerical_stabilty
/// [`Modified`]: Modified
/// [`GramSchmidt`]: GramSchmidt
/// [`GramSchmidt::compute`]: trait.GramSchmidt.html#tymethod.compute
pub fn mgs<S>(a: &ArrayBase<S, Ix2>) -> Result<(Array<f64, Ix2>, Array<f64, Ix2>)>
    where S: Data<Elem=f64>
{
    let mut mgs = Modified::from_matrix(a)?;
    mgs.compute(a)?;
    Ok((mgs.q().clone(), mgs.r().clone()))
}

#[cfg(test)]
generate_tests!(Modified, 1e-13);
