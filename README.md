# Gram Schmidt Orthonormalizatoin

Orthogonalization and QR decomposition of matrices in the Rust programming language and `rust-ndarray`.

This crate provides the following methods:

+ Classical Gram Schmidt, `cgs`,
+ Modified Gram Schmidt, `mgs`,
+ Classical Gram Schmidt with Reorthogonalization, `cgs2`.

# Usage

```rust
extern crate gramschmidt;
extern crate ndarray;

// Import openblas_src or another blas source to have the linker find all symbols.
extern crate openblas_src;

fn main() {
    let small_matrix = arr2(
        &[[2.0, 0.5, 0.0, 0.0],
          [0.0, 0.3, 0.0, 0.0],
          [0.0, 1.0, 0.7, 0.0],
          [0.0, 0.0, 0.0, 3.0]]
    );
    let mut cgs2 = ReorthogonalizedGramSchmidt::from_matrix(&small_matrix);
    cgs2.compute(&small_matrix);
    assert!(small_matrix.all_close(&cgs2.q().dot(cgs2.r()), 1e-14));
}
```

# Recent versions

+ `0.4.0`: Major rework of the library structure:
    + The algorithms are now configured via structs, the traits are dropped.
    + Provide the structs `ClassicalGramSchmidt`, `ModifiedGramSchmidt`, and
  `ReorthogonalizedGramSchmidt` (known as `cgs`, `mgs`, and `cgs2` in the
  literature, respectively);
    + `cgs` and `cgs2` are implemented using `blas` routines (major speedup!);
    + All routines are now able to handle column-major (Fortran-) and row-major (C-) order
    of the input matrices;
    + Remove parallel code.
+ `0.3.1`: Update to `blas 0.16` and do not specify a default backend (so that the user can set it).
+ `0.3.0`: Update to `ndarray 0.10`, `ndarray-parallel 0.5`
+ `0.2.1`: Added a parallelized algorithm using `rayon`
+ `0.2.0`: Update to `ndarray 0.9`
