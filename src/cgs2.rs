use cblas;
use ndarray::{
    Data,
    IntoDimension,
};
use ndarray::prelude::*;
use std::slice;

#[derive(Clone, Debug)]
pub struct ReorthogonalizedGramSchmidt {
    q: Array2<f64>,
    r: Array2<f64>,
    work_vector: Array1<f64>,
    memory_order: cblas::Layout,
    dirty: bool,
}

impl ReorthogonalizedGramSchmidt {
    /// Reserves the memory for a QR decomposition via a classical, reorthogonalized Gram Schmidt
    /// orthogonalization using the dimensions of a sample matrix. This is also known as CGS2 in
    /// the literature. See [Giraud et al.] for a definition.
    ///
    /// The resulting object can be used to orthogonalize matrices of the same dimensions.
    ///
    /// [Giraud et al.]: https://doi.org/10.1007/s00211-005-0615-4
    ///
    /// # Example
    ///
    /// ```
    /// extern crate gram_schmidt;
    /// extern crate ndarray;
    /// extern crate ndarray_rand;
    /// extern crate rand;
    ///
    /// use gram_schmidt::ReorthogonalizedGramSchmidt;
    /// use ndarray::Array2;
    /// use ndarray_rand::RandomExt;
    /// use rand::distributions::Normal;
    ///
    /// # fn main() {
    ///
    /// let matrix = Array2::random((10,10), Normal::new(0.0, 1.0));
    /// let mut cgs2 = ReorthogonalizedGramSchmidt::from_matrix(&matrix);
    ///
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

        ReorthogonalizedGramSchmidt {
            q: Array2::zeros(array_shape),
            r: Array2::zeros(array_shape),
            work_vector: Array1::zeros(a.dim().0),
            memory_order,
            dirty: false,
        }
    }

    /// Reserves the memory for a QR decomposition via a classical, reorthogonalized Gram Schmidt
    /// orthogonalization using a shape.
    ///
    /// The resulting object can be used to orthogonalize matrices of the same dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// extern crate gram_schmidt;
    ///
    /// use gram_schmidt::ReorthogonalizedGramSchmidt;
    ///
    /// # fn main() {
    /// let fortran_order = false;
    /// let mut cgs2 = ReorthogonalizedGramSchmidt::from_shape((10,10), fortran_order);
    ///
    /// # }
    /// ```
    pub fn from_shape<T>(shape: T, fortran_order: bool) -> Self
        where T: IntoDimension<Dim = Ix2>,
    {
        let dimension = shape.into_dimension();
        let rows = dimension.into_pattern().0;
        let array_shape = dimension.set_f(fortran_order);
        let memory_order = if fortran_order {
            cblas::Layout::ColumnMajor
        } else {
            cblas::Layout::RowMajor
        };

        ReorthogonalizedGramSchmidt {
            q: Array2::zeros(array_shape),
            r: Array2::zeros(array_shape),
            work_vector: Array1::zeros(rows),
            memory_order,
            dirty: false,
        }
    }

    /// Computes a QR decomposition using the classical, reorthogonalized Gram Schmidt
    /// orthonormalization of the matrix `a`.
    ///
    /// The input matrix `a` has to have exactly the same dimension and memory layout as was
    /// previously configured. Panics otherwise.
    ///
    /// ```
    /// extern crate gram_schmidt;
    /// extern crate ndarray;
    /// extern crate ndarray_rand;
    /// extern crate openblas_src;
    /// extern crate rand;
    ///
    /// use gram_schmidt::ReorthogonalizedGramSchmidt;
    /// use ndarray::Array2;
    /// use ndarray_rand::RandomExt;
    /// use rand::distributions::Normal;
    ///
    /// # fn main() {
    ///
    /// let matrix = Array2::random((10,10), Normal::new(0.0, 1.0));
    /// let mut cgs2 = ReorthogonalizedGramSchmidt::from_matrix(&matrix);
    /// cgs2.compute(&matrix);
    ///
    /// # }
    /// ```
    pub fn compute<S>(&mut self, a: &ArrayBase<S, Ix2>)
        where S: Data<Elem = f64>,
    {
        assert_eq!(a.shape(), self.q.shape());
        // assert_eq!(a.strides(), self.q.strides());

        let (n_rows, n_cols) = self.q.dim();

        let (a_column_inc, a_layout) = if let Some(_) = a.as_slice() {
            (n_cols as i32, cblas::Layout::RowMajor)
        } else if let Some(_) = a.as_slice_memory_order() {
            (1, cblas::Layout::RowMajor)
        } else {
            panic!("Array not contiguous!")
        };

        // If the struct was previously used for a computation, r needs to be cleaned.
        // q is always overwritten, but r is not.
        if self.dirty {
            self.r.fill(0.0);
        }

        for i in 0..n_cols {
            self.q.column_mut(i).assign(&a.column(i));
            let a_column = match a_layout {
                cblas::Layout::RowMajor => &a.as_slice_memory_order().unwrap()[i..],
                cblas::Layout::ColumnMajor => &a.as_slice_memory_order().unwrap()[n_rows * i..],
            };

            if i > 0 {
                let (lda, r_column, r_inc) = match self.memory_order {
                    cblas::Layout::RowMajor => {
                        let r_column = &mut self.r.as_slice_memory_order_mut().unwrap()[i..];
                        (n_cols as i32, r_column, n_cols as i32)
                    },

                    cblas::Layout::ColumnMajor => {
                        let r_column = &mut self.r.as_slice_memory_order_mut().unwrap()[n_rows*i..];
                        (n_rows as i32, r_column, 1)
                    },
                };

                // The unsafe blocks below are because we need several overlapping slices into the
                // q matrix. The mutable `q_column` is the i-th orthogonal vector which is currently
                // being sought. The immutable `q` represents the already found vectors (vectors 0
                // to i-1). Appropriately choosing the offset, lda, and increment makes sure that
                // blas does not access the same memory location.
                //
                // This is only necessary for row major formats. In column major formats there
                // won't be overlaps.
                let len = self.q.len();
                let q_ptr = self.q.as_mut_ptr();
                let q = unsafe {
                    slice::from_raw_parts(q_ptr, len)
                };
                let (q_column, q_inc) = match self.memory_order {
                    cblas::Layout::RowMajor => {
                        let offset = i as isize;
                        let q_column = unsafe {
                            slice::from_raw_parts_mut(q_ptr.offset(offset), len - i)
                        };
                        (q_column, n_cols as i32)
                    },

                    cblas::Layout::ColumnMajor => {
                        let offset = n_rows * i;
                        let q_column = unsafe {
                            slice::from_raw_parts_mut(q_ptr.offset(offset as isize), len - offset)
                        };
                        (q_column, 1)
                    },
                };

                let work_vector = self.work_vector.as_slice_memory_order_mut().unwrap();

                unsafe {
                    cblas::dgemv(
                        self.memory_order, // layout
                        cblas::Transpose::Ordinary, // transa
                        n_rows as i32, // m
                        i as i32, // n
                        1.0, // alpha
                        q, // a
                        lda, // lda
                        a_column, // x
                        a_column_inc, // incx
                        0.0, // beta
                        r_column, //y
                        r_inc // incy
                    );

                    cblas::dgemv(
                        self.memory_order, // layout
                        cblas::Transpose::None, // transa
                        n_rows as i32, // m
                        i as i32, // n
                        -1.0, // alpha
                        q, // a
                        lda, // lda
                        r_column, // y
                        r_inc, // incx
                        1.0, // beta
                        q_column, // y
                        q_inc, // incy
                    );

                    cblas::dgemv(
                        self.memory_order, // layout
                        cblas::Transpose::Ordinary, // transa
                        n_rows as i32, // m
                        i as i32, // n
                        1.0, // alpha
                        q, // a
                        lda, // lda
                        q_column, // x
                        q_inc, // incx
                        0.0, // beta
                        work_vector, //y
                        1 // incy
                    );

                    cblas::dgemv(
                        self.memory_order, // layout
                        cblas::Transpose::None, // transa
                        n_rows as i32, // m
                        i as i32, // n
                        -1.0, // alpha
                        q, // a
                        lda, // lda
                        work_vector, // y
                        1, // incx
                        1.0, // beta
                        q_column, // y
                        q_inc, // incy
                    );

                    cblas::daxpy(
                        n_rows as i32, // n
                        1.0, // alpha
                        work_vector, // x
                        1, // incx
                        r_column, // y
                        r_inc, // incy
                    );

                }
            };

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

            let mut v = self.q.column_mut(i);
            v /= norm;
            self.r[(i,i)] = a.column(i).dot(&v);
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

#[cfg(test)]
mod tests {
    use ndarray::prelude::*;
    use super::*;
    #[test]
    fn unity_is_unity() {
        let a = arr2(&[[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]]);

        let mut cgs2 = ReorthogonalizedGramSchmidt::from_matrix(&a);
        cgs2.compute(&a);

        assert_eq!(a, cgs2.q().dot(cgs2.r()));
    }
}
