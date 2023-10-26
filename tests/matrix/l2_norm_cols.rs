use custos::prelude::Float;

#[cfg(test)]
#[cfg(feature = "cpu")]
#[test]
fn test_l2_norm_cols() {
    use sliced::{test_utils::roughly_equals, Matrix, CPU};

    let device = CPU::<custos::Base>::new();

    let lhs = Matrix::from((&device, 2, 4, [1., 2., 3., 4., 5., 6., 7., 8.]));
    let out = lhs.l2_norm_cols::<()>();
    println!("out: {:?}", out.read());

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let expected_grad = [0.1826, 0.3651, 0.5477, 0.7303];

        roughly_equals(&expected_grad, &*lhs.grad());
    }
}
