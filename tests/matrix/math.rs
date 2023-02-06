use custos::CPU;
use sliced::Matrix;

#[test]
fn test_ops() {
    let device = CPU::new();

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

    let out: Matrix = lhs.gemm(&rhs);
    println!("out: {:?}", out.read());
}
