#[test]
fn test_softmax_with_matrix() {
    use sliced::{Matrix, CPU};

    let device = CPU::new();

    let x = Matrix::from((&device, 2, 3, [1., 2., 3., 4., 5., 6.]));
    let out = x.softmax();

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let expected_grad = [0.; 6];

        sliced::test_utils::roughly_equals(&expected_grad, &*x.grad());
    }
}
