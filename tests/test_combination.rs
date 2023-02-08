use custos::{get_count, range, ApplyFunction, Buffer, Combiner, OpenCL, TapeReturn, CPU};
use sliced::{BinaryElementWise, BinaryOpsMayGrad, SquareMayGrad};

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

        rawsliced::ew_assign_scalar(&mut x_grad, &0.1, |x, r| *x *= r);
        rawsliced::ew_assign_binary(&mut x, &x_grad, |x, y| *x -= y);
    }
}

#[test]
fn test_perf_min_this() {
    //let device = OpenCL::new(0).unwrap();
    let device = CPU::new();

    let x = Buffer::from((&device, [1.3f32; 123412]));
    let start = std::time::Instant::now();

    const TIMES: usize = 100;

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

    println!("sliced: {:?}", start.elapsed() / TIMES as u32);
}

#[test]
fn test_2perf_min_this() {
    //let device = OpenCL::new(0).unwrap();
    let device = CPU::new();

    let x: Buffer = Buffer::from((&device, vec![1.3f32; 1293412]));
    let b = Buffer::from((&device, vec![2.1f32; 1293412]));
    let start = std::time::Instant::now();

    const TIMES: usize = 40;

    for _ in 0..TIMES {
        for _ in range(100) {
            x.grad().clear();
            b.grad().clear();

            let squared = device.square(&x);
            let mul = device.mul(&squared, &x);
            let add = device.add(&b, &x);
            let mul_b = device.mul(&add, &b);
            let out = device.mul(&mul, &mul_b);
            //assert_eq!(&*out, [(1.3 * 1.3 * 1.3) * (1.3 + 2.1) * 2.1; 123412]);
            //println!("count: {}", get_count());

            //let _sum = out.iter().sum::<f32>();
            //println!("i: {i}, sum: {sum:?}");

            out.backward();

            //sliced::ew_assign_scalar(&mut x_grad, &0.1, |x, r| *x *= r);
            //sliced::ew_assign_binary(&mut x, &x_grad, |x, y| *x -= y);
        }
    }

    println!("elapsed: {:?}", start.elapsed() / TIMES as u32);
}

// default hasher: 101.5us
// seahash: 83us
// FxHash: 49us
#[test]
fn test_small_2perf_min_this() {
    //let device = OpenCL::new(0).unwrap();
    let device = CPU::new();

    let x = Buffer::from((&device, [1.3f32; 100]));
    let b = Buffer::from((&device, [2.1f32; 100]));
    let start = std::time::Instant::now();

    const TIMES: usize = 10000;

    for _ in 0..TIMES {
        for _ in range(100) {
            x.grad().clear();
            b.grad().clear();

            let squared = device.square(&x);
            let mul = device.mul(&squared, &x);
            let add = device.add(&b, &x);
            let mul_b = device.mul(&add, &b);
            let out = device.mul(&mul, &mul_b);
            //assert_eq!(&*out, [(1.3 * 1.3 * 1.3) * (1.3 + 2.1) * 2.1; 123412]);

            let _sum = out.iter().sum::<f32>();
            //println!("i: {i}, sum: {sum:?}");

            out.backward();

            //sliced::ew_assign_scalar(&mut x_grad, &0.1, |x, r| *x *= r);
            //sliced::ew_assign_binary(&mut x, &x_grad, |x, y| *x -= y);
        }
    }

    println!("elapsed: {:?}", start.elapsed() / TIMES as u32);
}
