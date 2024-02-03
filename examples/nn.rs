use std::time::Instant;

use custos::{
    prelude::Float, Alloc, Autograd, Base, Buffer, Device, IsShapeIndep, MayTapeActions,
    OnNewBuffer, TapeActions, CPU,
};

use graplot::Plot;
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
        let mut weights = Matrix::new(device, I, O);
        // device.rand(&mut weights, T::from_f64(-0.1), T::from);
        device.rand(&mut weights, -T::one() / T::two(), T::one() / T::two());
        //let mut weights = Matrix::from((device, I, O, vec![T::one(); I*O]));

        Linear {
            weights,
            bias: Matrix::new(device, 1, O),
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

    (x, y)
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
use std::ops::{Deref, DerefMut};

#[cfg(feature = "autograd")]
impl<T: Copy + One + Mul<Output = T> + SubAssign + 'static> SGD<T> {
    pub fn zero_grad<D>(&self, params: Vec<Param<T, D>>)
    where
        D: MayTapeActions + WriteBuf<T> + Alloc<T> + ClearBuf<T> + 'static,
    {
        for param in params {
            param.param.grad_mut().clear();
        }
    }

    pub fn step<D>(&self, params: Vec<Param<T, D>>)
    where
        D: WriteBuf<T> + Alloc<T> + MayTapeActions + 'static,
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

fn mnist() {
    let device = CPU::<Autograd<Base>>::new();

    let mut lin1 = Linear::<f32, _, { 28 * 28 }, 128>::new(&device);
    let mut lin2 = Linear::<f32, _, 128, 10>::new(&device);
    let mut lin3 = Linear::<f32, _, 10, 10>::new(&device);

    let loader = purpur::CSVLoader::new(true);
    let Ok(loaded_data) =
        loader.load::<f32, _>("../gradients-fallback/datasets/digit-recognizer/train.csv")
    else {
        return;
    };

    let mut x = Matrix::from((&device, loaded_data.sample_count, 28 * 28, loaded_data.x));
    for i in 0..x.len() {
        x[i] /= 255.;
    }

    let y = Matrix::from((&device, loaded_data.sample_count, 28 * 28, loaded_data.y));

    let sgd = SGD { lr: 0.0001 };
    for epoch in 0..50 {
        let out = lin1.forward(&x).relu();
        let out = lin2.forward(&out).relu();
        let out = lin3.forward(&out).softmax();

        let loss = (&out - &y).pow(2.);

        let avg_loss = device.mean(&loss);
        println!("epoch: {epoch}, loss: {avg_loss}");

        loss.backward();

        sgd.step(lin1.params());
        sgd.step(lin2.params());
        sgd.step(lin3.params());
    }
}

fn main() {
    mnist();
    return;
    let device = CPU::<Autograd<custos::Base>>::new();
    // let mut device = custos::OpenCL::<custos::Base>::new(0).unwrap();
    // device.set_unified_mem(false);

    // let mut lin1 = Linear::<f32, _, 1, 64>::new(&device);
    // let mut lin2 = Linear::<f32, _, 64, 64>::new(&device);
    // let mut lin3 = Linear::<f32, _, 64, 1>::new(&device);

    let mut lin1 = Linear::<f32, _, 1, 512>::new(&device);
    let mut lin2 = Linear::<f32, _, 512, 512>::new(&device);
    let mut lin3 = Linear::<f32, _, 512, 512>::new(&device);
    let mut lin4 = Linear::<f32, _, 512, 512>::new(&device);
    let mut lin5 = Linear::<f32, _, 512, 512>::new(&device);
    let mut lin6 = Linear::<f32, _, 512, 512>::new(&device);
    let mut lin7 = Linear::<f32, _, 512, 512>::new(&device);
    let mut lin8 = Linear::<f32, _, 512, 1>::new(&device);

    let (x, y) = create_sine(&device, 0, 100000);
    let sgd = SGD { lr: 0.0001 };

    let start = Instant::now();

    let mut already = false;

    for i in 0..18000 {
        #[cfg(feature = "autograd")]
        unsafe {
            device.gradients_mut().unwrap().zero_grad();
        };

        // sgd.zero_grad(lin1.params());
        // sgd.zero_grad(lin2.params());
        // sgd.zero_grad(lin3.params());

        // let out = lin1.forward(&x).relu();
        // let out = lin2.forward(&out).relu();
        // let out = lin3.forward(&out);

        let out = lin1.forward(&x).relu();
        let out = lin2.forward(&out).relu();
        let out = lin3.forward(&out).relu();
        let out = lin4.forward(&out).relu();
        let out = lin5.forward(&out).relu();
        let out = lin6.forward(&out).relu();
        let out = lin7.forward(&out).relu();
        let out = lin8.forward(&out);

        let loss = (&out - &y).pow(2.);

        println!("i: {i}");

        if !already {
            // println!("traces: {:?}", device.graph().cache_traces());
            // device.optimize().unwrap();
            already = true;
        }

        #[cfg(feature = "autograd")]
        {
            loss.backward();

            //println!("out: {:?}", &out.read_to_vec()[out.len()-100..]);
            //println!("lin1 dweights grad: {:?}", lin1.weights.grad().read_to_vec());
            sgd.step(lin1.params());
            sgd.step(lin2.params());
            sgd.step(lin3.params());
        }
    }

    println!("elapsed: {:?}", start.elapsed());

    let out = lin1.forward(&x).relu();
    let out = lin2.forward(&out).relu();
    let out = lin3.forward(&out);
    //println!("out: {:?}", out.read());

    let mut plot = Plot::new((x.read(), y.read()));
    plot.add((x.read(), out.read(), "-r"));
    plot.show()
}
