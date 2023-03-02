use custos::prelude::Float;

pub fn roughly_equals<T: Float>(lhs: &[T], rhs: &[T]) {
    for (a, b) in lhs.iter().zip(rhs) {
        let abs = (*a - *b).abs();
        if abs > T::one() / T::from_u64(100) {
            panic!(
                "\n left: '{:?}',\n right: '{:?}', \n left elem.: {} != right elem. {}",
                lhs, rhs, a, b
            )
        }
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_l2_norm_cols() {
    use sliced::{Matrix, CPU};

    let device = CPU::new();

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
