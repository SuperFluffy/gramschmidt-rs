use cblas;
use ndarray::{
    Data,
    IntoDimension,
};
use ndarray::prelude::*;
use std::slice;

#[derive(Clone, Debug)]
pub struct ClassicalGramSchmidt {
    q: Array2<f64>,
    r: Array2<f64>,
}

impl ClassicalGramSchmidt {
    /// Reserves the memory for a QR decomposition via a classical Gram Schmidt orthogonalization
    /// using the dimensions of a sample matrix.
    ///
    /// The resulting object can be used to orthogonalize matrices of the same dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// extern crate gram_schmidt;
    /// extern crate ndarray;
    /// extern crate ndarray_rand;
    /// extern crate rand;
    ///
    /// use gram_schmidt::ClassicalGramSchmidt;
    /// use ndarray::Array2;
    /// use ndarray_rand::RandomExt;
    /// use rand::distributions::Normal;
    ///
    /// let matrix = Array2::random((10,10), Normal::new(0.0, 1.0));
    /// let mut cgs = ClassicalGramSchmidt::from_matrix(&matrix);
    /// ```
    pub fn from_matrix<S>(a: &ArrayBase<S, Ix2>) -> Self
        where S: Data<Elem = f64>
    {
        ClassicalGramSchmidt {
            q: a.to_owned(),
            r: a.to_owned(),
        }
    }

    /// Reserves the memory for a QR decomposition via a classical Gram Schmidt orthogonalization
    /// using a shape.
    ///
    /// The resulting object can be used to orthogonalize matrices of the same dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// extern crate gram_schmidt;
    ///
    /// use gram_schmidt::ClassicalGramSchmidt;
    ///
    /// let mut cgs = ClassicalGramSchmidt::from_shape((10,10));
    /// ```
    pub fn from_shape<T>(shape: T) -> Self
        where T: IntoDimension<Dim = Ix2>,
    {
        let dimension = shape.into_dimension();
        ClassicalGramSchmidt {
            q: Array2::zeros(dimension),
            r: Array2::zeros(dimension),
        }
    }

    /// Computes a QR decomposition using the classical Gram Schmidt orthonormalization of the
    /// matrix `a`.
    ///
    /// The input matrix `a` has to have exactly the same dimension and memory layout as was
    /// previously configured. Panics otherwise.
    ///
    /// ```
    /// extern crate gram_schmidt;
    /// extern crate ndarray;
    /// extern crate ndarray_rand;
    /// extern crate rand;
    ///
    /// use gram_schmidt::ClassicalGramSchmidt;
    /// use ndarray::Array2;
    /// use ndarray_rand::RandomExt;
    /// use rand::distributions::Normal;
    ///
    /// let matrix = Array2::random((10,10), Normal::new(0.0, 1.0));
    /// let mut cgs = ClassicalGramSchmidt::from_matrix(&matrix);
    /// ```
    pub fn compute<S>(&mut self, a: &ArrayBase<S, Ix2>)
        where S: Data<Elem = f64>,
    {
        assert_eq!(a.shape(), self.q.shape());
        assert_eq!(a.strides(), self.q.strides());

        let layout = if let Some(_) = a.as_slice() {
            cblas::Layout::RowMajor
        } else if let Some(_) = a.as_slice_memory_order() {
            cblas::Layout::ColumnMajor
        } else {
            panic!("Array not contiguous!")
        };

        let (n_rows, n_cols) = self.q.dim();
        for i in 0..n_cols {
            println!("{}", i);
            self.q.column_mut(i).assign(&a.column(i));
            let refvec = &a.as_slice_memory_order().unwrap()[i..];
            let inc_refvec = a.strides()[0] as i32;

            if i > 0 {
                println!("Creating r");
                let (lda, r_column, r_inc) = match layout {
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
                let (q_column, q_inc) = match layout {
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

                unsafe {
                    cblas::dgemv(
                        layout, // layout
                        cblas::Transpose::Ordinary, // transa
                        n_rows as i32, // m
                        i as i32, // n
                        1.0, // alpha
                        q, // a
                        lda, // lda
                        refvec, // x
                        inc_refvec, // incx
                        0.0, // beta
                        r_column, //y
                        r_inc // incy
                    );

                    cblas::dgemv(
                        layout, // layout
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
                }
            };

            let norm = {
                let len = self.q.len();
                let q_ptr = self.q.as_mut_ptr();
                unsafe {
                    let (q_column, q_inc) = match layout {
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
}
