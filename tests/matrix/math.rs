use custos::CPU;
use sliced::Matrix;

#[test]
#[cfg_attr(miri, ignore)]
fn test_ops() {
    let device = CPU::<custos::Autograd<custos::Base>>::new();

    #[rustfmt::skip]
    let lhs = Matrix::from((&device, 2, 3,
        [1., 2., 3., 
        4., 5., 6.]
    ));

    #[rustfmt::skip]
    let rhs = Matrix::from((&device, 3, 2, 
        [1., 2., 
        3., 4., 
        5., 6.]
    ));

    let out: Matrix<_, _> = lhs.gemm(&rhs);
    println!("out: {:?}", out.read());
}
