extern crate cblas;
extern crate ndarray;

#[cfg(feature="parallel")]
extern crate ndarray_parallel;

#[cfg(test)]
extern crate rand;
#[cfg(test)]
extern crate ndarray_rand;
#[cfg(test)]
extern crate openblas_src;

#[cfg(feature="parallel")]
mod parallel;

mod cgs;
mod cgs2;
mod traits;
mod trait_impls;
mod utils;

#[cfg(feature="parallel")]
pub use parallel::ParallelModifiedGramSchmidt;

pub use cgs::ClassicalGramSchmidt;
pub use cgs2::ReorthogonalizedGramSchmidt;
pub use traits::ModifiedGramSchmidt;
pub use utils::*;

#[cfg(test)]
mod tests {
    use ndarray::{arr1,arr2,Array1,Array2};
    use super::ModifiedGramSchmidt;

    #[test]
    fn normalization() {
        let a = &[1.0,2.0,3.0,4.0,5.0];

        let n_manual = f64::sqrt(1.0 + 4.0 + 9.0 + 16.0 + 25.0);
        let n_calculated = super::normalization(a);

        println!("Norm, manual: {}", n_manual);
        println!("Norm, function: {}", n_calculated);

        assert_eq!(n_manual, n_calculated);
    }

    #[test]
    fn projection() {
        let a = arr1(&[1.0,2.0,3.0,4.0,5.0]);
        let p = super::project(&a,&a);

        println!("Projection: {}", p);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn classical_gram_schmidt() {
        let a = arr2(&[[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]]);

        let (a_orth, a_norm) = GramSchmidt::compute(&a);

        let b = arr2(&[[2.0, 0.5, 0.0, 0.0],
                       [0.0, 0.3, 0.0, 0.0],
                       [0.0, 1.0, 0.7, 0.0],
                       [0.0, 0.0, 0.0, 3.0]]);

        let (b_orth, b_norm) = GramSchmidt::compute(&b);

        println!("{:?}", b_orth);
        println!("{:?}", b_norm);

        assert!(super::orthogonal(&a_orth, 1e-14));
        assert!(a_norm.all_close(&Array1::from_elem(4, 1.0), 1e-16));

        assert!(super::orthogonal(&b_orth, 1e-14));
        assert!(b_norm.all_close(&arr1(&[2.0615528128088303, 0.2910427500435996, 0.7, 3.0]), 1e-16));
    }

    #[test]
    fn modified_gram_schmidt() {
        let a = arr2(&[[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]]);

        let (a_orth, a_norm) = ModifiedGramSchmidt::compute(&a);

        let b = arr2(&[[2.0, 0.5, 0.0, 0.0],
                       [0.0, 0.3, 0.0, 0.0],
                       [0.0, 1.0, 0.7, 0.0],
                       [0.0, 0.0, 0.0, 3.0]]);

        let (b_orth, b_norm) = ModifiedGramSchmidt::compute(&b);

        assert!(super::orthogonal(&a_orth,1e-14));
        assert!(a_norm.all_close(&Array1::from_elem(4, 1.0), 1e-16));

        assert!(super::orthogonal(&b_orth,1e-14));
        assert!(b_norm.all_close(&arr1(&[2.0615528128088303, 0.2910427500435996, 0.7, 3.0]), 1e-16));
    }

    #[test]
    fn modified_gram_schmidt_into() {
        let a = arr2(&[[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]]);

        let mut a_o = Array2::default((4,4));
        let mut a_n = Array1::default(4);

        let b = arr2(&[[2.0, 0.5, 0.0, 0.0],
                       [0.0, 0.3, 0.0, 0.0],
                       [0.0, 1.0, 0.7, 0.0],
                       [0.0, 0.0, 0.0, 3.0]]);

        let mut b_o = Array2::default((4,4));
        let mut b_n = Array1::default(4);

        let c = arr2(&[[-4.079764601288893, 4.831491499921403, -2.9560001027996132, -0.02239325297550033, -0.2672544204261703, -0.07718850306444144],
                       [1.2917480323712418, 0.030479388871438983, 0.604549448561548, 0.013409783846041783, 0.037439247530467186, 0.03153579130305008],
                       [-47.584641085515464, 5.501371846864031, 41.39822251681311, -33.69079455346558, 43.13388644338738, 68.7695035292409],
                       [2.5268795799504997, 25.418530275775225, 33.473125141381374, 77.3391516894698, -44.091836957161426, 45.10932299622911],
                       [-20.383209804181938, -19.163209972229616, 0.09795435026201423, -53.296988576627484, -88.482334971421, 16.757575995918756],
                       [62.270964677492124, -75.82678462673792, -0.6889077708993588, 2.2569901796884064, 9.21906803233946, 44.891962279862234]]);

        let mut c_o = Array2::default((6,6));
        let mut c_n = Array1::default(6);

        ModifiedGramSchmidt::compute_into(&a, &mut a_o, &mut a_n);
        ModifiedGramSchmidt::compute_into(&b, &mut b_o, &mut b_n);
        ModifiedGramSchmidt::compute_into(&c, &mut c_o, &mut c_n);

        assert!(super::orthogonal(&a_o,1e-14));
        assert!(a_n.all_close(&Array1::from_elem(4, 1.0), 1e-16));

        assert!(super::orthogonal(&b_o,1e-14));
        assert!(b_n.all_close(&arr1(&[2.0615528128088303, 0.2910427500435996, 0.7, 3.0]), 1e-16));

        assert!(super::orthogonal(&c_o,1e-14));
    }

    #[cfg(feature="parallel")]
    #[test]
    fn sequential_equals_parallel() {

        use ndarray_rand::RandomExt;
        use rand;
        use super::ParallelModifiedGramSchmidt;


        let size = 256;
        let dist = rand::distributions::Normal::new(0.0, 1.0);
        let matrix = Array2::random([size,size], dist);

        let (orth_seq, norm_seq) = ModifiedGramSchmidt::compute(&matrix);
        let (orth_par, norm_par) = ParallelModifiedGramSchmidt::compute(&matrix);

        assert!(orth_seq.all_close(&orth_par, 1e-16));
        assert!(norm_seq.all_close(&norm_par, 1e-16));
    }
}
