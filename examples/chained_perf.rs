use std::{
    hint::black_box,
    ops::{Add, Mul, DerefMut, Deref},
};

use custos::{
    Alloc, Combiner, Device, Dim1, Resolve, Shape,
    WithShape, Retriever, Base, Cached, Autograd,
};
use sliced::{slice_binary_ew, BinaryOpsMayGrad, Buffer, SquareMayGrad, CPU};

pub fn op<'b, T, D, S>(
    lhs: &Buffer<'b, T, D, S>,
    rhs: &Buffer<T, D, S>,
    op: impl Fn(T, T) -> T,
) -> Buffer<'b, T, D, S>
where
    D: Alloc<T> + Retriever<T>,
    D::Data<T, S>: Deref<Target = [T]> + DerefMut,
    S: Shape,
    T: Copy,
{
    let mut out = lhs.device().retrieve(lhs.len(), (lhs, rhs));

    for ((lhs, rhs), out) in lhs.iter().zip(&*rhs).zip(&mut out) {
        *out = op(*lhs, *rhs)
    }

    out
}

pub fn slice_binary_ew2<T>(lhs: &[T], rhs: &[T], out: &mut [T], f: impl Fn(T, T) -> T)
where
    T: Copy + Add<Output = T>,
{
    for i in 0..out.len() {
        out[i] = lhs[i] + rhs[i]
    }
    /*for ((lhs, rhs), out) in lhs.iter().zip(rhs.iter()).zip(out.iter_mut()) {
        *out = f(*lhs, *rhs)
    }*/
}

// 123412
// 12.8 MB 13.1MB
// dur: 23.82ms

// 1413412
// 80MB, 110 MB
// dur: 434 ms
fn main() {
    // let device = custos::OpenCL::<custos::Base>::new(0).unwrap();
    // let device = custos::Stack;
    let device = CPU::<Autograd<Cached<Base>>>::new();
    //device.tape_mut().disable();

    const SIZE: usize = 2; // 123412

    // if with fails with CPU -> backwards operation may use () shape, Box<dyn Any> does not like this
    let mut x = Buffer::with(&device, [1.3f32; SIZE]);
    let mut b = Buffer::with(&device, [2.1f32; SIZE]);

    let start = std::time::Instant::now();

    const TIMES: usize = 1;

    let mut already = false;

    /*let x_slice = &*x;
    let b_slice = &*b;*/
    for _ in 0..TIMES {
        for epoch in 0..100 {
            //let mut out = device.retrieve::<_, Dim1<SIZE>>(SIZE, (&x, &b));
            /*for idx in 0..SIZE {
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
            assert_eq!(out.read()[0], 9.336999);

            out.backward();

            // let squared = device.mul(&x, &x);
            // let mut squared: Buffer<'_, f32, _, Dim1<SIZE>> = device.retrieve(x.len(), (&x, &x));
            // slice_binary_ew2(&x, &x, &mut squared, |x, _| x.mul(x));

            // let mut add: Buffer<'_, f32, _, Dim1<SIZE>> = device.retrieve(b.len(), (&b, &x));
            // slice_binary_ew2(&b, &x, &mut add, |b, x| b.add(x));

            // let mut mul_b: Buffer<'_, f32, _, Dim1<SIZE>> = device.retrieve(add.len(), (&add, &b));
            // slice_binary_ew2(&add, &b, &mut mul_b, |add, b| add.mul(b));

            // let mut mul: Buffer<'_, f32, _, Dim1<SIZE>> = device.retrieve(squared.len(), (&squared, &x));
            // slice_binary_ew2(&squared, &x, &mut mul, |squared, x| squared.mul(x));

            // let mut out: Buffer<'_, f32, _, Dim1<SIZE>> = device.retrieve(mul.len(), (&mul, &mul_b));
            // slice_binary_ew2(&mul, &mul_b, &mut out, |mul, mul_b| mul.add(mul_b));

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
        // println!("next")
    }

    println!(
        "elapsed (custos/sliced): {:?}",
        start.elapsed() / TIMES as u32
    );
}
