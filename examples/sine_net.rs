use custos::{
    prelude::Float, AddOperation, Alloc, AsNoId, Autograd, Base, Buffer, Cached, Cursor, Device,
    ExecNow, IsShapeIndep, Lazy, MayTapeActions, OnNewBuffer, Run, TapeActions, ZeroGrad,
};

use sliced::{GemmMayGrad, Matrix, Mean, RandOp, RowOpMayGrad};

pub struct Linear<'a, T, D: Device, const I: usize, const O: usize> {
    weights: Matrix<'a, T, D>,
    bias: Matrix<'a, T, D>,
}

impl<'a, T: Float, D: Device + OnNewBuffer<T, D>, const I: usize, const O: usize>
    Linear<'a, T, D, I, O>
{
    pub fn new(device: &'a D) -> Self
    where
        D: RandOp<T> + Alloc<T>,
    {
        let mut weights = Matrix::new(device, I, O).require_grad();
        // device.rand(&mut weights, T::from_f64(-0.1), T::from);
        device.rand(&mut weights, -T::one() / T::two(), T::one() / T::two());
        //let mut weights = Matrix::from((device, I, O, vec![T::one(); I*O]));

        Linear {
            weights,
            bias: Matrix::new(device, 1, O).require_grad(),
        }
    }

    #[inline]
    pub fn forward(&self, inputs: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        D: GemmMayGrad<T> + RowOpMayGrad<T>,
    {
        //inputs.gemm(&self.weights).add_row(&self.bias)
        let mut out = inputs.gemm(&self.weights);
        out.add_row_mut(&self.bias);
        out
    }

    pub fn params<'b>(&'b mut self) -> Vec<Param<'b, 'a, T, D>>
    where
        D: IsShapeIndep,
    {
        vec![Param::new(&mut self.weights), Param::new(&mut self.bias)]
    }
}

pub fn create_sine<'a, D: Alloc<f32> + IsShapeIndep + OnNewBuffer<f32, D>>(
    device: &'a D,
    min: usize,
    max: usize,
) -> (Matrix<f32, D>, Matrix<f32, D>) {
    let mut x: Vec<f32> = Vec::new();
    for add in min..max {
        x.push(add as f32 / 1000.);
    }
    let y = x
        .iter()
        .map(|v| (2. * v * std::f32::consts::PI).sin())
        .collect::<Vec<f32>>();

    let x = Matrix::from((device, max - min, 1, x));
    let y = Matrix::from((device, max - min, 1, y));

    (x.no_grad(), y.no_grad())
}

pub struct Param<'a, 'b, T, D: Device> {
    param: &'a mut Buffer<'b, T, D>,
}

impl<'a, 'b, T, D: Device> Param<'a, 'b, T, D> {
    pub fn new(param: &'a mut Buffer<'b, T, D>) -> Self {
        Param { param }
    }
}

pub struct SGD<T> {
    lr: T,
}

#[cfg(feature = "autograd")]
use custos::prelude::{ClearBuf, One, WriteBuf};

#[cfg(feature = "autograd")]
use core::ops::{Mul, SubAssign};
use std::{
    ops::{Deref, DerefMut},
    rc::Rc,
};

#[cfg(feature = "autograd")]
impl<T: Copy + One + Mul<Output = T> + SubAssign + 'static> SGD<T> {
    pub fn zero_grad<D>(&self, params: Vec<Param<T, D>>)
    where
        D: MayTapeActions + WriteBuf<T> + Alloc<T> + ZeroGrad<T> + ClearBuf<T> + 'static,
    {
        for param in params {
            param.param.grad_mut().clear();
        }
    }

    pub fn step<D>(&self, params: Vec<Param<T, D>>)
    where
        D: WriteBuf<T> + Alloc<T> + MayTapeActions + ZeroGrad<T> + 'static,
        D::Base<T, ()>: Deref<Target = [T]> + DerefMut,
    {
        for param in params {
            let grad = param.param.grad();
            for (value, grad) in param.param.iter_mut().zip(grad.iter()) {
                *value -= *grad * self.lr
            }
        }
    }
}

fn sine_net() {
    use std::time::Instant;

    use custos::CPU;

    let dev = CPU::<Autograd<Cached<Base>>>::new();
    let mut lin1 = Linear::<f32, _, 1, 64>::new(&dev);
    let mut lin2 = Linear::<f32, _, 64, 64>::new(&dev);
    let mut lin3 = Linear::<f32, _, 64, 1>::new(&dev);

    let (x, y) = create_sine(&dev, 0, 1000);

    let sgd = SGD { lr: 0.0001 };

    let start = Instant::now();

    for _ in dev.range(0..10000) {
        #[cfg(feature = "autograd")]
        unsafe {
            dev.gradients_mut().unwrap().zero_grad();
        };

        // custos::TapeReturn::tape_mut(&device).grads.zero_grad();
        // sgd.zero_grad(lin1.params());
        // sgd.zero_grad(lin2.params());
        // sgd.zero_grad(lin3.params());

        let out = lin1.forward(&x).relu();
        let out = lin2.forward(&out).relu();
        let out = lin3.forward(&out);

        let loss = (&out - &y).pow(2.);
        let loss_val = dev.mean(&loss);
        // println!("loss: {loss_val}");

        #[cfg(feature = "autograd")]
        {
            loss.backward();

            // println!("lin1 dweights grad: {:?}", lin1.weights.grad());

            sgd.step(lin1.params());
            sgd.step(lin2.params());
            sgd.step(lin3.params());
        }
    }

    println!("elapsed: {:?}", start.elapsed());

    let out = lin1.forward(&x).relu();
    let out = lin2.forward(&out).relu();
    let out = lin3.forward(&out);
    // println!("out: {:?}", out.read());

    let mut plot = graplot::Plot::new((x.read(), y.read()));
    plot.add((x.read(), out.read(), "-r"));
    plot.show()
}

fn sine_net_lazy2() {
    use std::time::Instant;

    use custos::CPU;

    let dev = Rc::new(CPU::<Autograd<Lazy<Base>>>::new());
    let mut lin1 = Linear::<f32, _, 1, 64>::new(&*dev);
    let mut lin2 = Linear::<f32, _, 64, 64>::new(&*dev);
    let mut lin3 = Linear::<f32, _, 64, 1>::new(&*dev);

    let (x, y) = create_sine(&*dev, 0, 1000);
    let sgd = SGD { lr: 0.0001 };

    let start = Instant::now();

    dev.clone()
        .add_op(dev.clone().no_id(), |dev| {
            #[cfg(feature = "autograd")]
            unsafe {
                dev.gradients_mut().unwrap().zero_grad();
            };
            Ok(())
        })
        .unwrap();

    let out = lin1.forward(&x).relu();
    let out = lin2.forward(&out).relu();
    let out = lin3.forward(&out);

    let loss = (&out - &y).pow(2.);

    for _ in 0..10000 {
        unsafe { dev.run().unwrap() };
        let loss_val = dev.mean(loss.replace());
        // println!("loss: {loss_val}");

        #[cfg(feature = "autograd")]
        {
            loss.replace().backward();

            sgd.step(lin1.params());
            sgd.step(lin2.params());
            sgd.step(lin3.params());
        }
    }
    println!("elapsed: {:?}", start.elapsed());

    let out = lin1.forward(&x).relu();
    let out = lin2.forward(&out).relu();
    let out = lin3.forward(&out);
    unsafe { dev.exec_now(&dev, ..).unwrap() };

    let mut plot = graplot::Plot::new((x.read(), y.read()));
    plot.add((x.read(), out.replace().read(), "-r"));
    plot.show()
}

fn sine_net_lazy() {
    use std::time::Instant;

    use custos::CPU;

    let dev = Rc::new(CPU::<Autograd<Lazy<Base>>>::new());
    let mut lin1 = Linear::<f32, _, 1, 64>::new(&*dev);
    let mut lin2 = Linear::<f32, _, 64, 64>::new(&*dev);
    let mut lin3 = Linear::<f32, _, 64, 1>::new(&*dev);

    let (x, y) = create_sine(&*dev, 0, 1000);

    let sgd = SGD { lr: 0.0001 };

    let start = Instant::now();

    for _ in dev.range(0..1000) {
        dev.clone()
            .add_op(dev.clone().no_id(), |dev| {
                #[cfg(feature = "autograd")]
                unsafe {
                    dev.gradients_mut().unwrap().zero_grad();
                };
                Ok(())
            })
            .unwrap();

        let out = lin1.forward(&x).relu();
        let out = lin2.forward(&out).relu();
        let out = lin3.forward(&out);

        let loss = (&out - &y).pow(2.);
        unsafe { dev.exec_now(&dev, ..).unwrap() };
        let loss_val = dev.mean(loss.replace());
        println!("loss: {loss_val}");

        #[cfg(feature = "autograd")]
        {
            loss.replace().backward();

            sgd.step(lin1.params());
            sgd.step(lin2.params());
            sgd.step(lin3.params());
        }
    }

    println!("elapsed: {:?}", start.elapsed());

    let out = lin1.forward(&x).relu();
    let out = lin2.forward(&out).relu();
    let out = lin3.forward(&out);
    unsafe { dev.exec_now(&dev, ..).unwrap() };

    let mut plot = graplot::Plot::new((x.read(), y.read()));
    plot.add((x.read(), out.replace().read(), "-r"));
    plot.show()
}

fn main() {
    sine_net_lazy2();
    sine_net();
}
