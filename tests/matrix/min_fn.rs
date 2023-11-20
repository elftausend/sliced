#[cfg(feature = "autograd")]
#[test]
fn test_min_fn() {
    use custos::{Autograd, Base, Buffer, CPU};
    use sliced::Matrix;

    let device = CPU::<Autograd<Base>>::new();
    let mut x = Matrix::from((&device, 1, 7, [10f32, -10., 10., -5., 6., 3., 1.]));

    for i in 0..100 {
        // add powi
        let squared = x.pow(2.);

        println!("i: {:?}, sum: {:?}", i, squared.iter().sum::<f32>());

        squared.backward();

        let mut grad = x.grad_mut();
        rawsliced::ew_assign_scalar(&mut grad, &0.1, |x, r| *x *= r);
        rawsliced::ew_assign_binary(&mut x, &grad, |x, y| *x -= y);
        grad.clear();
    }
}
