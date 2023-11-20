#[cfg(feature = "cpu")]
#[test]
fn test_tanh_cpu() {
    use custos::{Autograd, CPU};
    use sliced::{test_utils::roughly_equals, Matrix};

    let device = CPU::<Autograd<custos::Base>>::new();

    let buf = Matrix::from((&device, 1, 5, [-1f32, -3., 2., 5., -1.3, 0.]));
    let out = buf.tanh();

    roughly_equals(
        out.read(),
        &[-0.7615942, -0.9950548, 0.9640276, 0.9999092, -0.8617231, 0.],
    );

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let grad = buf.grad();
        roughly_equals(
            grad.read(),
            &[
                0.41997433,
                0.009865999,
                0.070650816,
                0.00018155575,
                0.25743324,
                1.0,
            ],
        );
    }
}
