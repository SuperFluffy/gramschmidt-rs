extern crate cblas;
extern crate ndarray;

#[cfg(feature="parallel")]
extern crate ndarray_parallel;

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

#[cfg(test)]
extern crate ndarray_rand;

#[cfg(test)]
extern crate openblas_src;

#[cfg(test)]
extern crate rand;

#[cfg(feature="parallel")]
mod parallel;

mod cgs;
mod cgs2;
mod mgs;

#[cfg(test)]
pub(crate) mod utils;

#[cfg(feature="parallel")]
pub use parallel::ParallelModifiedGramSchmidt;

pub use cgs::ClassicalGramSchmidt;
pub use cgs2::ReorthogonalizedGramSchmidt;
pub use mgs::ModifiedGramSchmidt;

// #[cfg(test)]
// mod tests {
    // #[cfg(feature="parallel")]
    // #[test]
    // fn sequential_equals_parallel() {

    //     use ndarray_rand::RandomExt;
    //     use rand;
    //     use super::ParallelModifiedGramSchmidt;


    //     let size = 256;
    //     let dist = rand::distributions::Normal::new(0.0, 1.0);
    //     let matrix = Array2::random([size,size], dist);

    //     let (orth_seq, norm_seq) = ModifiedGramSchmidt::compute(&matrix);
    //     let (orth_par, norm_par) = ParallelModifiedGramSchmidt::compute(&matrix);

    //     assert!(orth_seq.all_close(&orth_par, 1e-16));
    //     assert!(norm_seq.all_close(&norm_par, 1e-16));
    // }
// }
