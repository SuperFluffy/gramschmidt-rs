extern crate openblas_src;

use gramschmidt::{
    cgs,
    cgs2,
    mgs
};
use lapacke::dlange;
use ndarray::{
    Array2,
    ArrayBase,
    Data,
    Ix2,
    s,
};

fn create_lauchli(n: usize, epsilon: f64) -> Array2<f64>
{
    let mut matrix = Array2::<f64>::zeros((n+1, n));
    matrix
        .row_mut(0)
        .fill(1.0);
    matrix
        .slice_mut(s![1.., ..])
        .diag_mut()
        .fill(epsilon);
    matrix
}

fn two_norm<S>(matrix: &ArrayBase<S, Ix2>) -> f64
where
    S: Data<Elem=f64>,
{
    let (n_rows, n_cols) = matrix.dim();
    let slice = matrix.as_slice_memory_order().unwrap();

    unsafe {
        dlange(
        lapacke::Layout::RowMajor,
        b'F',
        n_rows as i32,
        n_cols as i32,
        slice,
        n_cols as i32,
    )}

}

fn main()
{
    // let epsilon = std::f64::EPSILON.sqrt();
    let epsilon = 0.0001;

    let lauchli_matrix = create_lauchli(3, epsilon);

    let (q_cgs, r_cgs) = cgs(&lauchli_matrix).expect("Failed to perform CGS");
    let (q_cgs_repeated, r_cgs_repeated) = cgs(&q_cgs).expect("Failed to perform CGS a second time");
    let (q_cgs2, r_cgs2) = cgs2(&lauchli_matrix).expect("Failed to perform CGS2");
    let (q_mgs, r_mgs) = mgs(&lauchli_matrix).expect("Failed to perform MGS");

    let unity = Array2::<f64>::eye(3);

    println!("Epsilon used: {}\n", epsilon);

    println!("Lauchli matrix:\n{:?}\n", lauchli_matrix);

    println!("Results for cgs:");
    println!("Q:\n{:?}\n", q_cgs);
    println!("R:\n{:?}\n", r_cgs);
    println!("Q*R:\n{:?}\n", q_cgs.dot(&r_cgs));
    println!("QᵀQ\n{:?}\n", q_cgs.t().dot(&q_cgs));
    println!("𝟙 - QᵀQ\n{:?}\n", &unity - &(q_cgs.t().dot(&q_cgs)));
    println!("‖𝟙 - QᵀQ‖₂\n{:?}\n", two_norm(&(&unity - &(q_cgs.t().dot(&q_cgs)))));

    println!("Results for repeated cgs:");
    println!("Q':\n{:?}\n", q_cgs_repeated);
    println!("R':\n{:?}\n", r_cgs_repeated);
    println!("Q'*R':\n{:?}\n", q_cgs_repeated.dot(&r_cgs_repeated));
    println!("Q'ᵀQ'\n{:?}\n", q_cgs_repeated.t().dot(&q_cgs_repeated));
    println!("𝟙 - Q'ᵀQ'\n{:?}\n", &unity - &(q_cgs_repeated.t().dot(&q_cgs_repeated)));
    println!("‖𝟙 - Q'ᵀQ'‖₂\n{:?}\n", two_norm(&(&unity - &(q_cgs_repeated.t().dot(&q_cgs_repeated)))));

    println!("Results for cgs2:");
    println!("Q:\n{:?}\n", q_cgs2);
    println!("R:\n{:?}\n", r_cgs2);
    println!("Q*R:\n{:?}\n", q_cgs2.dot(&r_cgs2));
    println!("QᵀQ\n{:?}\n", q_cgs2.t().dot(&q_cgs2));
    println!("𝟙 - QᵀQ\n{:?}\n", &unity - &(q_cgs2.t().dot(&q_cgs2)));
    println!("‖𝟙 - QᵀQ‖₂\n{:?}\n", two_norm(&(&unity - &(q_cgs2.t().dot(&q_cgs2)))));

    println!("Results for mgs:");
    println!("Q:\n{:?}\n", q_mgs);
    println!("R:\n{:?}\n", r_mgs);
    println!("Q*R:\n{:?}\n", q_mgs.dot(&r_mgs));
    println!("QᵀQ\n{:?}\n", q_mgs.t().dot(&q_mgs));
    println!("𝟙 - QᵀQ\n{:?}\n", &unity - &(q_mgs.t().dot(&q_mgs)));
    println!("‖𝟙 - QᵀQ‖₂\n{:?}\n", two_norm(&(&unity - &(q_mgs.t().dot(&q_mgs)))));
}
