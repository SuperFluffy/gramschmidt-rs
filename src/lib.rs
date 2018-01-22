extern crate cblas;
extern crate ndarray;

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

#[cfg(test)]
extern crate ndarray_rand;

#[cfg(test)]
extern crate openblas_src;

#[cfg(test)]
extern crate rand;

mod cgs;
mod cgs2;
mod mgs;

#[cfg(test)]
pub(crate) mod utils;

pub use cgs::ClassicalGramSchmidt;
pub use cgs2::ReorthogonalizedGramSchmidt;
pub use mgs::ModifiedGramSchmidt;
