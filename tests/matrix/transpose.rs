use sliced::{BinaryOpsMayGrad, Buffer, Transpose};

#[test]
fn test_transpose_matrix() {
    use sliced::{Matrix, CPU};

    let device = CPU::<custos::Autograd<custos::Base>>::new();

    let x = Matrix::from((&device, 2, 3, [1., 2., 3., 4., 5., 6.]));
    let out = x.T::<()>();

    let y = Matrix::from((&device, 3, 2, [-1., 1., 2., -5., -3., 2.]));
    let out = device.mul(&y, &out);

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let expected_grad: Buffer<f64, _> = device.transpose(y.rows(), y.cols(), &y);

        sliced::test_utils::roughly_equals(&expected_grad, &*x.grad());
    }
}
