use custos::GenericBlas;
use sliced::slice_transpose;

#[test]
fn test_left_trans_gemm() {
    // 2 x 4;
    #[rustfmt::skip]
    let a = [
        1., 2., 3., 4.,
        4., 5., 6., 5.
    ];

    // 4 x 2
    let mut trans_a = [0.; 2 * 4];
    slice_transpose(2, 4, &a, &mut trans_a);

    // 2 x 3;
    #[rustfmt::skip]
    let b = [
        1., 2., 3.,
        4., 5., 6.
    ];

    let mut out = [0.; 4 * 3];

    GenericBlas::Tgemm(4, 3, 2, &a, &b, &mut out);
    println!("out: {out:?}");

    let mut t_before_out = [0.; 4 * 3];

    GenericBlas::gemm(4, 3, 2, &trans_a, &b, &mut t_before_out);
    println!("t_before out: {t_before_out:?}");

    assert_eq!(out, t_before_out);
}

#[test]
fn test_left_trans_gemm_large() {
    // 7 x 5
    #[rustfmt::skip]
    let a = [
        9., 1., 3., 6., 7., 3., 63., 93., 51., 23., 36., 87., 3., 63., 9., 1., 43., 46.3, 7., 3.,
        63., 9., 15., 73., 6.3, 7., 53., 63., 69., 1., 3., 6., 7., 43., 63.,
    ];

    // 5 x 7
    let mut trans_a = [0.; 7 * 5];
    slice_transpose(7, 5, &a, &mut trans_a);

    // 7 x 10
    let b = [
        1., 2., 3., 44., 55., 6., 7., 8., 95., 103., 14., 2., 33., 4., 75., 6., 37., 8., 9., 120.,
        31., 2., 3., 4., 5., 6.51, 7.45, 8., 9., 10., 313., 244., 3., 4., 5.8, 6., 27., 48., 9.,
        101., 21., 2., 3.4324, 4., 5., 6., 75., 38., 9., 109., 11., 2., 3., 4., 85., 96., 7., 8.,
        29., 130., 1., 2.91, 3.909, 4., 5.634, 36., 7., 8., 9., 130.,
    ];

    let mut out = [0.; 5 * 10];

    GenericBlas::Tgemm(5, 10, 7, &a, &b, &mut out);

    let mut t_before_out = [0.; 5 * 10];

    GenericBlas::gemm(5, 10, 7, &trans_a, &b, &mut t_before_out);

    assert_eq!(out, t_before_out);
}

#[test]
fn test_right_trans_gemm() {
    // 4 x 3;
    #[rustfmt::skip]
    let a = [
        1., 2., 3.,
        4., 5., 6.,
        7., 8., 9.,
        10., 11., 12.,
    ];

    // 2 x 3;
    #[rustfmt::skip]
    let b = [
        1., 2., 3.,
        4., 5., 6.
    ];

    // 3 x 2
    let mut trans_b = [0.; 2 * 3];
    slice_transpose(2, 3, &a, &mut trans_b);

    let mut out = [0.; 4 * 2];

    GenericBlas::gemmT(4, 2, 3, &a, &b, &mut out);
    println!("out: {out:?}");

    let mut t_before_out = [0.; 4 * 2];

    GenericBlas::gemm(4, 2, 3, &a, &trans_b, &mut t_before_out);
    println!("t_before out: {t_before_out:?}");

    assert_eq!(out, t_before_out);
}
