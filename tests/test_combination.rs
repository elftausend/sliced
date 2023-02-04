use adal::{BinaryOps, Square, BinaryElementWise};
use custos::{CPU, Buffer, range, ApplyFunction, Combiner};


#[test]
fn test_comb() {
    let device = CPU::new();

    let mut x = Buffer::from((&device, [10f32, -10., 10., -5., 6., 3., 1.]));

    for i in range(100) {
        let mut x_grad = x.grad();
        x_grad.clear();

        let squared = device.square(&x);

        let sum = squared.iter().sum::<f32>();
        println!("i: {i}, sum: {sum:?}");

        squared.backward();

        sliced::ew_assign_scalar(&mut x_grad, &0.1, |x, r| *x *= r);
        sliced::ew_assign_binary(&mut x, &x_grad, |x, y| *x -= y);
    }
}

#[test]
fn test_perf_min_this() {
    let device = CPU::new();

    let x = Buffer::from((&device, [1.3f32; 123412]));
    let start = std::time::Instant::now();

    const TIMES: usize = 1000;

    for _ in 0..TIMES {
        for _ in range(100) {
            let mut x_grad = x.grad();
            x_grad.clear();

            let squared = device.square(&x);
            let mul = device.mul(&squared, &x);
            assert_eq!(&*mul, [1.3 * 1.3 * 1.3; 123412]);

            let _sum = squared.iter().sum::<f32>();
            //println!("i: {i}, sum: {sum:?}");

            mul.backward();

            //sliced::ew_assign_scalar(&mut x_grad, &0.1, |x, r| *x *= r);
            //sliced::ew_assign_binary(&mut x, &x_grad, |x, y| *x -= y);
        }
    }

    println!("elapsed: {:?}", start.elapsed() / TIMES as u32);
}