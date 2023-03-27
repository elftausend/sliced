use std::hint::black_box;

use custos::{range, Dim1, WithShape};
use sliced::{BinaryOpsMayGrad, Buffer, SquareMayGrad, Stack, CPU};

fn main() {
    //let device = OpenCL::new(0).unwrap();
    // let device = Stack;
    let device = CPU::new();
    //device.tape_mut().disable();

    let mut x = Buffer::with(&device, [1.3f32; 123412]);
    let mut b = Buffer::from((&device, vec![2.1f32; 123412]));

    let start = std::time::Instant::now();

    const TIMES: usize = 100;

    for _ in 0..TIMES {
        for epoch in range(100) {
            let squared = device.square(&x);
            let mul = device.mul(&squared, &x);
            let add = device.add(&b, &x);
            let mul_b = device.mul(&add, &b);
            let out = device.add(&mul, &mul_b);

            if epoch % 200 == 0 {
                println!("out: {}", out.read()[0]);
            }
            //assert_eq!(&*out, [(1.3 * 1.3 * 1.3) * (1.3 + 2.1) * 2.1; 123412]);
            //println!("count: {}", get_count());

            //let _sum = out.iter().sum::<f32>();
            //println!("i: {i}, sum: {sum:?}");

            // out.backward();

            // x.grad_mut().clear();
            // b.grad_mut().clear();

            //sliced::ew_assign_scalar(&mut x_grad, &0.1, |x, r| *x *= r);
            //sliced::ew_assign_binary(&mut x, &x_grad, |x, y| *x -= y);
        }
    }

    println!("elapsed: {:?}", start.elapsed() / TIMES as u32);
}
