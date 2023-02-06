use std::time::Instant;
use custos::{
    prelude::{Float, Number},
    range, Alloc, Device, IsShapeIndep, CPU,
};
use sliced::{Gemm, GemmMayGrad, Matrix, RandOp, RowOpMayGrad};

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
        device.rand(&mut weights, -T::one(), T::one());

        Linear {
            weights,
            bias: Matrix::new(device, 1, O),
        }
    }

    pub fn forward(&self, inputs: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        D: GemmMayGrad<T> + RowOpMayGrad<T>,
    {
        inputs.gemm(&self.weights).add_row(&self.bias)
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

#[test]
fn test_nn() {
    let device = CPU::new();
    let lin1 = Linear::<f32, _, 1, 64>::new(&device);
    let lin2 = Linear::<f32, _, 64, 64>::new(&device);
    let lin3 = Linear::<f32, _, 64, 1>::new(&device);

    let (x, y) = create_sine(&device, 0, 1000);

    let start = Instant::now();

    for _ in range(1000) {
        let out = lin1.forward(&x).relu();
        let out = lin2.forward(&out).relu();
        let out = lin3.forward(&out);

        let loss = (&out - &y).squared();
        loss.backward();
    }

    println!("elapsed: {:?}", start.elapsed());
}
