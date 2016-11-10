extern crate blas;
#[macro_use(s)]
extern crate ndarray;

use ndarray::{Array1,Array2,Data,DataMut,Ix1,Ix2,LinalgScalar};
use ndarray::prelude::*;

pub trait GramSchmidt: Sized + Clone + Default {
    fn compute_into<S1,S2>(a: &ArrayBase<S1, Ix2>, orth: &mut ArrayBase<S2, Ix2>, norm: &mut ArrayBase<S2, Ix1>)
        where S1: Data<Elem = Self>,
              S2: DataMut<Elem = Self>;

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
    fn compute_into<S1,S2>(a: &ArrayBase<S1, Ix2>, orth: &mut ArrayBase<S2, Ix2>, norm: &mut ArrayBase<S2, Ix1>)
        where S1: Data<Elem = Self>,
              S2: DataMut<Elem = Self>;

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
    fn compute_into<S1,S2>(a: &ArrayBase<S1, Ix2>, orth: &mut ArrayBase<S2, Ix2>, norm: &mut ArrayBase<S2, Ix1>)
        where S1: Data<Elem = Self>,
              S2: DataMut<Elem = Self>
    {
        let n_rows = orth.shape()[0];

        orth.row_mut(0).assign(&a.row(0));

        for i in 0..n_rows {
            let (done, mut todo) = orth.view_mut().split_at(Axis(0), i);
            let ref_vec = a.row(i);

            let mut v = todo.row_mut(0);

            for w in done.inner_iter() {
                let projection = project(&ref_vec, &w);
                v.zip_mut_with(&w, |ev,ew| { *ev -= projection * ew; });
            }

            norm[i] = normalization(v.as_slice().unwrap());
            v /= norm[i];
        }
    }
}

impl ModifiedGramSchmidt for f64 {
    fn compute_into<S1,S2>(a: &ArrayBase<S1, Ix2>, orth: &mut ArrayBase<S2, Ix2>, norm: &mut ArrayBase<S2, Ix1>)
        where S1: Data<Elem = Self>,
              S2: DataMut<Elem = Self>
    {
        let n_rows = orth.shape()[0];

        orth.row_mut(0).assign(&a.row(0));

        for i in 0..n_rows {
            let (done, mut todo) = orth.view_mut().split_at(Axis(0), i);

            let mut v = todo.row_mut(0);

            for w in done.inner_iter() {
                let projection = project(&v, &w);
                v.zip_mut_with(&w, |ev,ew| { *ev -= projection * ew; });
            }

            norm[i] = normalization(v.as_slice().unwrap());
            v /= norm[i];
        }
    }
}

pub fn orthogonal<S>(a: &ArrayBase<S,Ix2>) -> bool
    where S: Data<Elem=f64>
{
    let b = a.dot(&a.t());
    b.all_close(&Array2::eye(b.shape()[0]), 1e-14)
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
    use ndarray::{arr1,arr2,Array1};
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

        assert!(super::orthogonal(&a_orth));
        assert!(a_norm.all_close(&Array1::from_elem(4, 1.0), 1e-16));

        assert!(super::orthogonal(&b_orth));
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

        println!("{:?}", b_orth);
        println!("{:?}", b_norm);

        assert!(super::orthogonal(&a_orth));
        assert!(a_norm.all_close(&Array1::from_elem(4, 1.0), 1e-16));

        assert!(super::orthogonal(&b_orth));
        assert!(b_norm.all_close(&arr1(&[2.0615528128088303, 0.2910427500435996, 0.7, 3.0]), 1e-16));
    }
}
