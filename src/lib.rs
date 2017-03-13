extern crate blas;
extern crate ndarray;

use ndarray::{Data,DataMut,LinalgScalar};
use ndarray::prelude::*;

pub trait GramSchmidt: Sized + Clone + Default {
    fn compute_into<S1,S2,S3>(a: &ArrayBase<S1, Ix2>, orth: &mut ArrayBase<S2, Ix2>, norm: &mut ArrayBase<S3, Ix1>)
        where S1: Data<Elem = Self>,
              S2: DataMut<Elem = Self>,
              S3: DataMut<Elem = Self>;

    fn compute<S>(a: &ArrayBase<S, Ix2>) -> (Array<Self, Ix2>, Array<Self, Ix1>)
        where S: Data<Elem = Self>
    {
        let mut o = a.to_owned();
        let mut n = Array1::default(a.dim().1);

        Self::compute_into(a, &mut o, &mut n);
        (o, n)
    }
}

pub trait ModifiedGramSchmidt: Sized + Clone + Default {
    fn compute_inplace<S1,S2>(orth: &mut ArrayBase<S1, Ix2>, norm: &mut ArrayBase<S2, Ix1>)
        where S1: DataMut<Elem = Self>,
              S2: DataMut<Elem = Self>;

    fn compute_inplace_no_norm<S1>(a: &mut ArrayBase<S1, Ix2>)
        where S1: DataMut<Elem = Self>;

    fn compute_into<S1,S2,S3>(a: &ArrayBase<S1, Ix2>, orth: &mut ArrayBase<S2, Ix2>, norm: &mut ArrayBase<S3, Ix1>)
        where S1: Data<Elem = Self>,
              S2: DataMut<Elem = Self>,
              S3: DataMut<Elem = Self>,
    {
        orth.assign(&a);
        Self::compute_inplace(orth, norm);
    }

    fn compute<S>(a: &ArrayBase<S, Ix2>) -> (Array<Self, Ix2>, Array<Self, Ix1>)
        where S: Data<Elem = Self>
    {
        let mut o = a.to_owned();
        let mut n = Array1::default(a.dim().1);

        Self::compute_into(a, &mut o, &mut n);
        (o, n)
    }
}

impl GramSchmidt for f64 {
    fn compute_into<S1,S2,S3>(a: &ArrayBase<S1, Ix2>, orth: &mut ArrayBase<S2, Ix2>, norm: &mut ArrayBase<S3, Ix1>)
        where S1: Data<Elem = Self>,
              S2: DataMut<Elem = Self>,
              S3: DataMut<Elem = Self>
    {
        let n_rows = orth.shape()[0];

        orth.row_mut(0).assign(&a.row(0));

        for i in 0..n_rows {
            let (done, mut todo) = orth.view_mut().split_at(Axis(0), i);
            let ref_vec = a.row(i);

            let mut v = todo.row_mut(0);

            for w in done.inner_iter() {
                let projection_factor = project(&ref_vec, &w);
                v.zip_mut_with(&w, |ev,ew| { *ev -= projection_factor * ew; });
            }

            norm[i] = normalization(v.as_slice().unwrap());
            v /= norm[i];
        }
    }
}

impl ModifiedGramSchmidt for f64 {
    fn compute_inplace<S1,S2>(orth: &mut ArrayBase<S1, Ix2>, norm: &mut ArrayBase<S2, Ix1>)
        where S1: DataMut<Elem = Self>,
              S2: DataMut<Elem = Self>,
    {
        let n_rows = orth.shape()[0];

        // Orthonormalize the current row with respect to all already orthonormalized rows.
        //
        // Another strategy would have been to normalize the current row, and then remove it
        // from all not-yet-orthonormalized rows. However, benchmarking reveals that the first
        // strategy is about 10% faster.
        for i in 0..n_rows {
            let (done, mut todo) = orth.view_mut().split_at(Axis(0), i);

            let mut v = todo.row_mut(0);

            for w in done.inner_iter() {
                // w is already normalized
                // let projection_factor = project(&v, &w);
                let projection_factor = v.dot(&w);
                v.zip_mut_with(&w, |ev,ew| { *ev -= projection_factor * ew; });
            }

            norm[i] = normalization(v.as_slice().unwrap());
            v /= norm[i];
        }
    }

    fn compute_inplace_no_norm<S1>(orth: &mut ArrayBase<S1, Ix2>)
        where S1: DataMut<Elem = Self>,
    {
        let n_rows = orth.shape()[0];

        for i in 0..n_rows {
            let (done, mut todo) = orth.view_mut().split_at(Axis(0), i);

            let mut v = todo.row_mut(0);

            for w in done.inner_iter() {
                // w is already normalized
                // let projection_factor = project(&v, &w);
                let projection_factor = v.dot(&w);
                v.zip_mut_with(&w, |ev,ew| { *ev -= projection_factor * ew; });
            }

            v /= normalization(v.as_slice().unwrap());
        }
    }
}

pub fn orthogonal<S>(a: &ArrayBase<S,Ix2>, tol: f64) -> bool
    where S: Data<Elem=f64>
{
    let b = a.dot(&a.t());
    b.all_close(&Array2::eye(b.shape()[0]), tol)
}

pub fn normalization(v: &[f64]) -> f64 {
    blas::c::dnrm2(v.len() as i32, v, 1)
}

pub fn project<A,S1,S2>(vector: &ArrayBase<S1,Ix1>, onto: &ArrayBase<S2,Ix1>) -> A
    where S1: Data<Elem=A>,
          S2: Data<Elem=A>,
          A: LinalgScalar
{
    vector.dot(onto) / onto.dot(onto)
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1,arr2,Array1,Array2};
    use super::{GramSchmidt,ModifiedGramSchmidt};

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
    fn gram_schmidt() {
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
}
