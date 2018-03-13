use cblas;
use ndarray::{
    Data,
    IntoDimension,
};
use ndarray::prelude::*;
use std::slice;

#[derive(Clone, Debug)]
pub struct ModifiedGramSchmidt {
    q: Array2<f64>,
    r: Array2<f64>,
    memory_order: cblas::Layout,
}

impl ModifiedGramSchmidt {
    /// Reserves the memory for a QR decomposition via a modified Gram Schmidt orthogonalization
    /// using the dimensions of a sample matrix.
    ///
    /// The resulting object can be used to orthogonalize matrices of the same dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// extern crate gramschmidt;
    /// extern crate ndarray;
    /// extern crate ndarray_rand;
    /// extern crate rand;
    ///
    /// use gramschmidt::ModifiedGramSchmidt;
    /// use ndarray::Array2;
    /// use ndarray_rand::RandomExt;
    /// use rand::distributions::Normal;
    ///
    /// # fn main() {
    /// let matrix = Array2::random((10,10), Normal::new(0.0, 1.0));
    /// let mut cgs = ModifiedGramSchmidt::from_matrix(&matrix);
    /// # }
    /// ```
    pub fn from_matrix<S>(a: &ArrayBase<S, Ix2>) -> Self
        where S: Data<Elem = f64>
    {
        let (is_fortran, memory_order) = if let Some(_) = a.as_slice() {
            (false, cblas::Layout::RowMajor)
        } else if let Some(_) = a.as_slice_memory_order() {
            (true, cblas::Layout::ColumnMajor)
        } else {
            panic!("Array not contiguous!")
        };

        let array_shape = a.raw_dim().set_f(is_fortran);
        ModifiedGramSchmidt {
            q: Array2::zeros(array_shape),
            r: Array2::zeros(array_shape),
            memory_order,
        }
    }

    /// Reserves the memory for a QR decomposition via a modified Gram Schmidt orthogonalization
    /// using a shape.
    ///
    /// The resulting object can be used to orthogonalize matrices of the same dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// extern crate gramschmidt;
    ///
    /// use gramschmidt::ModifiedGramSchmidt;
    ///
    /// # fn main() {
    /// let fortran_order = false;
    /// let mut cgs = ModifiedGramSchmidt::from_shape((10,10), fortran_order);
    /// # }
    /// ```
    pub fn from_shape<T>(shape: T, fortran_order: bool) -> Self
        where T: IntoDimension<Dim = Ix2>,
    {
        let dimension = shape.into_dimension();
        let array_shape = dimension.set_f(fortran_order);
        let memory_order = if fortran_order {
            cblas::Layout::ColumnMajor
        } else {
            cblas::Layout::RowMajor
        };
        ModifiedGramSchmidt {
            q: Array2::zeros(array_shape),
            r: Array2::zeros(array_shape),
            memory_order,
        }
    }

    /// Computes a QR decomposition using the modified Gram Schmidt orthonormalization of the
    /// matrix `a`.
    ///
    /// The input matrix `a` has to have exactly the same dimension and memory layout as was
    /// previously configured. Panics otherwise.
    ///
    /// ```
    /// extern crate gramschmidt;
    /// extern crate ndarray;
    /// extern crate ndarray_rand;
    /// extern crate rand;
    ///
    /// use gramschmidt::ModifiedGramSchmidt;
    /// use ndarray::Array2;
    /// use ndarray_rand::RandomExt;
    /// use rand::distributions::Normal;
    ///
    /// # fn main() {
    /// let matrix = Array2::random((10,10), Normal::new(0.0, 1.0));
    /// let mut cgs = ModifiedGramSchmidt::from_matrix(&matrix);
    /// # }
    /// ```
    pub fn compute<S>(&mut self, a: &ArrayBase<S, Ix2>)
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
                    let (q_column, q_inc) = match self.memory_order {
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
    }

    /// Return a reference to the matrix q.
    pub fn q(&self) -> &Array2<f64> {
        &self.q
    }

    /// Return a reference to the matrix q.
    pub fn r(&self) -> &Array2<f64> {
        &self.r
    }
}

generate_tests!(ModifiedGramSchmidt, 1e-13);
