use ndarray::prelude::*;
use ndarray::{Data,DataMut};

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
