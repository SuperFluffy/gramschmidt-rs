macro_rules! generate_tests {
    ($method:ident, $tolerance:expr) => {
        #[cfg(test)]
        mod tests {
            extern crate openblas_src;

            use lazy_static::lazy_static;
            use ndarray::prelude::*;
            use super::*;

            lazy_static!(
                static ref UNITY: Array2<f64> = arr2(
                    &[[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]]
                );
            );

            lazy_static!(
                static ref F_UNITY: Array2<f64> =
                    Array2::from_shape_fn(
                        (4,4).f(),
                        |(i,j)| if i == j { 1.0 } else { 0.0 }
                    );
            );

            lazy_static!(
                static ref SMALL: Array2<f64> = arr2(
                    &[[2.0, 0.5, 0.0, 0.0],
                      [0.0, 0.3, 0.0, 0.0],
                      [0.0, 1.0, 0.7, 0.0],
                      [0.0, 0.0, 0.0, 3.0]]
                );
            );

            lazy_static!(
                static ref F_SMALL: Array2<f64> = 
                    Array2::from_shape_vec(
                        (4,4).f(),
                        vec![2.0, 0.0, 0.0, 0.0,
                             0.5, 0.3, 1.0, 0.0,
                             0.0, 0.0, 0.7, 0.0,
                             0.0, 0.0, 0.0, 3.0
                        ]
                    ).unwrap();
            );

            lazy_static!(
                static ref LARGE: Array2<f64> = arr2(
                    &[[-4.079764601288893, 4.831491499921403, -2.9560001027996132, -0.02239325297550033, -0.2672544204261703, -0.07718850306444144],
                      [1.2917480323712418, 0.030479388871438983, 0.604549448561548, 0.013409783846041783, 0.037439247530467186, 0.03153579130305008],
                      [-47.584641085515464, 5.501371846864031, 41.39822251681311, -33.69079455346558, 43.13388644338738, 68.7695035292409],
                      [2.5268795799504997, 25.418530275775225, 33.473125141381374, 77.3391516894698, -44.091836957161426, 45.10932299622911],
                      [-20.383209804181938, -19.163209972229616, 0.09795435026201423, -53.296988576627484, -88.482334971421, 16.757575995918756],
                      [62.270964677492124, -75.82678462673792, -0.6889077708993588, 2.2569901796884064, 9.21906803233946, 44.891962279862234]]
                );
            );

            lazy_static!(
                static ref F_LARGE: Array2<f64> = Array2::from_shape_vec(
                    (6,6).f(),
                    vec![-4.079764601288893, 4.831491499921403, -2.9560001027996132, -0.02239325297550033, -0.2672544204261703, -0.07718850306444144,
                         1.2917480323712418, 0.030479388871438983, 0.604549448561548, 0.013409783846041783, 0.037439247530467186, 0.03153579130305008,
                         -47.584641085515464, 5.501371846864031, 41.39822251681311, -33.69079455346558, 43.13388644338738, 68.7695035292409,
                         2.5268795799504997, 25.418530275775225, 33.473125141381374, 77.3391516894698, -44.091836957161426, 45.10932299622911,
                         -20.383209804181938, -19.163209972229616, 0.09795435026201423, -53.296988576627484, -88.482334971421, 16.757575995918756,
                         62.270964677492124, -75.82678462673792, -0.6889077708993588, 2.2569901796884064, 9.21906803233946, 44.891962279862234
                    ]
                ).unwrap();
            );

            #[test]
            fn unity_stays_unity() {
                let mut method = $method::from_matrix(&*UNITY);
                method.compute(&*UNITY);

                assert_eq!(&*UNITY, &method.q().dot(method.r()));
            }

            #[test]
            fn small_orthogonal() {
                let mut method = $method::from_matrix(&*SMALL);
                method.compute(&*SMALL);
                assert!(crate::utils::orthogonal(method.q(),$tolerance));
            }

            #[test]
            fn small_qr_returns_original() {
                let mut method = $method::from_matrix(&*SMALL);
                method.compute(&*SMALL);
                assert!(SMALL.all_close(&method.q().dot(method.r()), $tolerance));
            }

            #[test]
            fn large_orthogonal() {
                let mut method = $method::from_matrix(&*LARGE);
                method.compute(&*LARGE);
                assert!(crate::utils::orthogonal(method.q(),$tolerance));
            }

            #[test]
            fn large_qr_returns_original() {
                let mut method = $method::from_matrix(&*LARGE);
                method.compute(&*LARGE);
                assert!(LARGE.all_close(&method.q().dot(method.r()), $tolerance));
            }

            #[test]
            fn f_order_unity_stays_unity() {
                let mut method = $method::from_matrix(&*F_UNITY);
                method.compute(&*F_UNITY);

                assert_eq!(&*F_UNITY, &method.q().dot(method.r()));
            }

            #[test]
            fn f_order_small_orthogonal() {
                let mut method = $method::from_matrix(&*F_SMALL);
                method.compute(&*F_SMALL);
                assert!(crate::utils::orthogonal(method.q(),$tolerance));
            }

            #[test]
            fn f_order_small_qr_returns_original() {
                let mut method = $method::from_matrix(&*F_SMALL);
                method.compute(&*F_SMALL);
                assert!(F_SMALL.all_close(&method.q().dot(method.r()), $tolerance));
            }

            #[test]
            fn f_order_large_orthogonal() {
                let mut method = $method::from_matrix(&*F_LARGE);
                method.compute(&*F_LARGE);
                assert!(crate::utils::orthogonal(method.q(),$tolerance));
            }

            #[test]
            fn f_order_large_qr_returns_original() {
                let mut method = $method::from_matrix(&*F_LARGE);
                method.compute(&*F_LARGE);
                assert!(F_LARGE.all_close(&method.q().dot(method.r()), $tolerance));
            }
        }
    }
}
