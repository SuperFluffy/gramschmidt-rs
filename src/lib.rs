//! # Gram Schmidt procedures for Rust and `ndarray`
//!
//! This crate implements three different Gram Schmidt procedures in the form of QR decompositions:
//!
//! + The [classical Gram Schmidt] procedure, `[cgs]`;
//! + the [modified or stabilized Gram Schmidt] procedure, `[mgs]`;
//! + the [reorthogonalized Gram Schmidt procedure], `[cgs2]`.
//!
//! [ndarray]: https://github.com/rust-ndarray/ndarray
//! [classical Gram Schmidt]: https://en.wikipedia.org/wiki/Gram-Schmidt_process
//! [modified or stabilized Gram Schmidt]: https://en.wikipedia.org/wiki/Gram-Schmidt_process#Numerical_stabilty
//! [reorthogonalized Gram Schmidt procedure]: https://doi.org/10.1007/s00211-005-0615-4
//! [cgs]: struct.Classical#method.cgs

use ndarray::{
    ArrayBase,
    Array2,
    Data,
    Dim,
    Ix,
    Ix2,
    ShapeBuilder,
};
use std::error;
use std::result;
use std::fmt;

#[cfg(test)]
#[macro_use]
mod test_macros;

mod cgs;
mod cgs2;
mod mgs;

pub(crate) mod utils;

// Reexports
pub use cgs::Classical;
pub use cgs2::Reorthogonalized;
pub use mgs:: Modified;

/// Errors that occur during a initialization of a Gram Schmidt factorization.
#[derive(Debug)]
pub enum Error {
    /// The layout of the matrix to be factorized is incompatible with the layout the GramSchmidt
    /// procedure was configured for. It means that the GramSchmidt procedure is configured to
    /// work with either column major (Fortran layout) or row major (C layout) matrices.
    IncompatibleLayouts,

    /// The array to be factorized is not contiguous. At the moment, all arrays to be factorized
    /// have to be contiguous.
    NonContiguous,
}

pub type Result<T> = result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Error::*;
        match self {
            IncompatibleLayouts => write!(f, "The arrays representing the matrices don't have the same layouts."),
            NonContiguous => write!(f, "Array shape is not contiguous"),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

pub trait GramSchmidt: Sized {
    /// Reserves the memory for a QR decomposition via a classical Gram Schmidt orthogonalization
    /// using a shape.
    ///
    /// The resulting object can be used to orthogonalize matrices of the same dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// use gramschmidt::{
    ///     Classical,
    ///     GramSchmidt,
    ///     Result,
    /// };
    ///
    /// # fn main() -> Result<()> {
    ///
    /// let mut cgs = Classical::from_shape((10,10))?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn from_shape<T>(shape: T) -> Result<Self>
        where T: ShapeBuilder<Dim = Dim<[Ix; 2]>>;

    /// Computes a QR decomposition using a Gram Schmidt orthonormalization of the matrix `a`.
    ///
    /// The input matrix `a` has to have exactly the same dimension and memory layout as was
    /// previously configured. Returns an error otherwise.
    ///
    /// ```
    /// extern crate openblas_src;
    ///
    /// use gramschmidt::{GramSchmidt, Classical};
    /// use ndarray::Array2;
    /// use ndarray_rand::RandomExt;
    /// use rand::distributions::Normal;
    ///
    /// # fn main() {
    ///
    /// let matrix = Array2::random((10,10), Normal::new(0.0, 1.0));
    /// let mut cgs = Classical::from_matrix(&matrix).unwrap();
    /// cgs.compute(&matrix);
    ///
    /// # }
    /// ```
    fn compute<S>(&mut self, a: &ArrayBase<S, Ix2>) -> Result<()>
        where S: Data<Elem = f64>;

    /// Return a reference to the matrix q.
    fn q(&self) -> &Array2<f64>;

    /// Return a reference to the matrix q.
    fn r(&self) -> &Array2<f64>;

    // Blanket impls
    /// One-off version of [`compute`]. Takes the matrix `a` to be factorized, allocates a type
    /// implementing the `GramSchmidt` trait, computes the QR decomposition, and returns clones of
    /// the Q and R matrices.
    ///
    /// [`compute`]: trait.GramSchmidt.html#method.compute
    fn compute_once<S>(a: &ArrayBase<S, Ix2>) -> Result<(Array2<f64>, Array2<f64>)>
        where S: Data<Elem=f64>,
    {
        let mut gram_schmidt = Self::from_matrix(a)?;
        gram_schmidt.compute(a)?;
        Ok((gram_schmidt.q().clone(), gram_schmidt.r().clone()))
    }

    /// Uses a matrix to reserve memory for a QR decomposition via a classical Gram Schmidt.
    ///
    /// The resulting object can be used to orthogonalize matrices of the same dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::Array;
    /// use gramschmidt::{
    ///     Classical,
    ///     GramSchmidt,
    ///     Result,
    /// };
    ///
    /// # fn main() -> Result<()> {
    ///
    /// let a = Array::zeros((10, 10));
    /// let mut cgs = Classical::from_matrix(&a)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn from_matrix<S>(a: &ArrayBase<S, Ix2>) -> Result<Self>
        where S: Data<Elem = f64>
    {
        use cblas::Layout::*;
        let dim = a.dim();
        let shape = match utils::get_layout(a) {
            Some(ColumnMajor) => dim.f(),
            Some(RowMajor) => dim.into_shape(),
            None => Err(Error::NonContiguous)?,
        };

        Self::from_shape(shape)
    }

}

/// Convenience function that calculates a [Classical Gram Schmidt] QR factorization, returning a
/// tuple `(Q,R)`.
///
/// If you want to repeatedly calculate QR factorizations, then prefer constructing a [`Classical`]
/// struct and calling its [`GramSchmidt::compute`] method implemented through the [`GramSchmidt`] trait.
///
/// [Classical Gram Schmidt]: https://en.wikipedia.org/wiki/Gram-Schmidt_process
/// [`Classical`]: Classical
/// [GramSchmidt]: GramSchmidt
/// [`GramSchmidt::compute`]: trait.GramSchmidt.html#tymethod.compute
pub fn cgs<S>(a: &ArrayBase<S, Ix2>) -> Result<(Array2<f64>, Array2<f64>)>
    where S: Data<Elem=f64>
{
    Classical::compute_once(a)
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
/// [`GramSchmidt::compute`]: trait.GramSchmidt.html#tymethod.compute
pub fn cgs2<S>(a: &ArrayBase<S, Ix2>) -> Result<(Array2<f64>, Array2<f64>)>
    where S: Data<Elem=f64>
{
    Reorthogonalized::compute_once(a)
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
pub fn mgs<S>(a: &ArrayBase<S, Ix2>) -> Result<(Array2<f64>, Array2<f64>)>
    where S: Data<Elem=f64>
{
    Modified::compute_once(a)
}
