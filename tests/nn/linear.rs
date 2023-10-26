use custos::{prelude::Float, Alloc, Buffer, Device, IsShapeIndep};

use sliced::{GemmMayGrad, Matrix, RandOp, RowOpMayGrad};

pub struct Linear<'a, T, D: Device, const I: usize, const O: usize> {
    weights: Matrix<'a, T, D>,
    bias: Matrix<'a, T, D>,
}

impl<'a, T: Float, D: Device, const I: usize, const O: usize> Linear<'a, T, D, I, O> {
    pub fn new(device: &'a D) -> Self
    where
        D: RandOp<T> + Alloc<'a, T>,
    {
        let mut weights = Matrix::new(device, I, O);
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

pub fn create_sine<'a, D: Alloc<'a, f32> + IsShapeIndep>(
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
use custos::prelude::{ClearBuf, MainMemory, MayTapeReturn, One, WriteBuf};

#[cfg(feature = "autograd")]
use core::ops::{Mul, SubAssign};

#[cfg(feature = "autograd")]
impl<T: Copy + One + Mul<Output = T> + SubAssign + 'static> SGD<T> {
    pub fn zero_grad<D>(&self, params: Vec<Param<T, D>>)
    where
        D: MayTapeReturn + WriteBuf<T> + for<'b> Alloc<'b, T> + ClearBuf<T> + 'static,
    {
        for param in params {
            param.param.grad_mut().clear();
        }
    }

    pub fn step<D>(&self, params: Vec<Param<T, D>>)
    where
        D: MainMemory + MayTapeReturn + WriteBuf<T> + for<'b> Alloc<'b, T> + 'static,
    {
        for param in params {
            let grad = param.param.grad_unbound();
            for (value, grad) in param.param.iter_mut().zip(grad.iter()) {
                *value -= *grad * self.lr
            }
        }
    }
}

#[cfg(feature = "autograd")]
#[test]
#[cfg_attr(miri, ignore)]
fn test_mnist() {
    use custos::CPU;
    use purpur::CSVLoader;

    let device = CPU::<custos::Base>::new();

    /*
    let loader = CSVLoader::new(true);
    let Ok(loaded_data) = loader.load::<f32, _>("../gradients-fallback/datasets/digit-recognizer/train.csv") else {
        return;
    };

    let mut lin1 = Linear::<f32, _, 1, 64>::new(&device);
    let mut lin2 = Linear::<f32, _, 64, 64>::new(&device);
    let mut lin3 = Linear::<f32, _, 64, 1>::new(&device);*/
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_nn() {
    use std::time::Instant;

    use custos::{range, CPU};

    let device = CPU::<custos::Base>::new();
    let mut lin1 = Linear::<f32, _, 1, 64>::new(&device);
    let mut lin2 = Linear::<f32, _, 64, 64>::new(&device);
    let mut lin3 = Linear::<f32, _, 64, 1>::new(&device);

    let (x, y) = create_sine(&device, 0, 1000);
    let sgd = SGD { lr: 0.0001 };

    let start = Instant::now();

    for _ in range(1000) {
        #[cfg(feature = "autograd")]
        custos::TapeReturn::tape_mut(&device).grads.zero_grad();
        // sgd.zero_grad(lin1.params());
        // sgd.zero_grad(lin2.params());
        // sgd.zero_grad(lin3.params());

        let out = lin1.forward(&x).relu();
        let out = lin2.forward(&out).relu();
        let out = lin3.forward(&out);

        let loss = (&out - &y).squared();

        #[cfg(feature = "autograd")]
        {
            loss.backward();

            //println!("lin1 dweights grad: {:?}", lin1.weights.grad());

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
    //plot.add((x.read(), out.read(), "-r"));
    //    plot.show()
}
