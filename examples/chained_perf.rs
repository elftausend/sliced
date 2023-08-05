use std::hint::black_box;

use custos::{
    range, Alloc, CacheReturn, Device, Dim1, GraphReturn, MainMemory, Resolve, Shape, WithShape,
};
use sliced::{BinaryOpsMayGrad, Buffer, SquareMayGrad, CPU};

pub fn op<'b, T, D, S>(
    lhs: &Buffer<'b, T, D, S>,
    rhs: &Buffer<T, D, S>,
    op: impl Fn(T, T) -> T,
) -> Buffer<'b, T, D, S>
where
    D: for<'a> Alloc<'a, T, S> + MainMemory,
    S: Shape,
    T: Copy,
{
    let mut out = lhs.device().retrieve(lhs.len(), (lhs, rhs));

    for ((lhs, rhs), out) in lhs.iter().zip(&*rhs).zip(&mut out) {
        *out = op(*lhs, *rhs)
    }

    out
}

// 123412
// 12.8 MB 13.1MB
// dur: 23.82ms

// 1413412
// 80MB, 110 MB
// dur: 434 ms
fn main() {
    //let device = OpenCL::new(0).unwrap();
    // let device = Stack;
    let device = CPU::new();
    //device.tape_mut().disable();

    const SIZE: usize = 2203412; // 123412

    // if with fails with CPU -> backwards operation may use () shape, Box<dyn Any> does not like this
    let mut x: Buffer = Buffer::from((&device, vec![1.3f32; SIZE]));
    let mut b = Buffer::from((&device, vec![2.1f32; SIZE]));

    let start = std::time::Instant::now();

    const TIMES: usize = 1000;

    let mut already = false;

    let x_slice = &*x;
    let b_slice = &*b;
    for _ in 0..TIMES {
        for epoch in range(100) {
            /*let mut out = device.retrieve::<_, Dim1<SIZE>>(SIZE, (&x, &b));
            for idx in 0..SIZE {
                let x = x_slice[idx];
                let b = b_slice[idx];

                let squared = x * x;
                let mul = squared * x;
                let add = b + x;
                let mul_b = add * b;
                out[idx] = mul + mul_b;
            }*/
            let squared = device.square(&x);
            let add = device.add(&b, &x);
            let mul_b = device.mul(&add, &b);
            let mul = device.mul(&squared, &x);
            let out = device.add(&mul, &mul_b);

            if epoch % 900 == 0 {
                println!("out: {}", out.read()[0]);
            }

            if !already {
                // println!("cache traces: {:?}", &device.graph().cache_traces());
                // println!("nodes before: {:?}", &device.cache().nodes);
                // device.optimize().unwrap();
                already = true;
                // println!();
                // println!("nodes after: {:?}", &device.cache().nodes);
            }
            // println!("{traces:?}");
            //let out = op(&x, &b, |x, b| x * x * x * (x + b) * b);

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
