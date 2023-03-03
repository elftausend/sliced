use std::fmt::Display;

use crate::{BinaryElementWise, Diagflat, Gemm, SoftmaxGrad, Transpose};
use custos::{range, Buffer, Device, GenericBlas, Shape, CPU, MainMemory};

impl<T, S> SoftmaxGrad<T, S> for CPU
where
    T: Copy + GenericBlas + Default + Display + core::ops::Sub<T, Output = T>,
    S: Shape,
    CPU: Gemm<T>,
{
    fn softmax_grad(
        &self,
        samples: usize,
        features: usize,
        x_grad: &mut Buffer<T, Self, S>,
        out: &Buffer<T, Self, S>,
        out_grad: &Buffer<T, Self, S>,
    ) {
        for idx in range(samples - 1) {
            let index = idx * features;

            // ensure that data is only read
            let single_out: Buffer<T> = unsafe {
                Buffer::from_raw_host((&out[index..index + features]).as_ptr() as *mut T, features)
            };

            let single_grad: Buffer<T> = unsafe {
                Buffer::from_raw_host(
                    (&out_grad[index..index + features]).as_ptr() as *mut T,
                    features,
                )
            };

            let diagflat: Buffer<T> = self.diagflat(&single_out);

            // cols 1 x 1 cols
            let jacobian_matrix: Buffer<T> = self.sub(
                &diagflat,
                &self.gemm(
                    features,
                    1,
                    features,
                    &single_out,
                    &self.transpose(features, 1, &single_out),
                ),
            );
            T::gemm(
                features,
                1,
                features,
                &jacobian_matrix,
                &single_grad,
                &mut x_grad[index..index + features],
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SoftmaxGrad;
    use custos::{range, Buffer, Device, GenericBlas, Shape, CPU, MainMemory};

    #[test]
    fn test_softmax_grad() {
        /*let device = CPU::new();
        let samples = 2;
        let features = 3;
        let mut x_grad = Buffer::<f32>::zeros(samples * features);
        let out = Buffer::<f32>::from_slice(&[0.1, 0.2, 0.7, 0.3, 0.4, 0.3]);
        let out_grad = Buffer::<f32>::from_slice(&[0.1, 0.2, 0.7, 0.3, 0.4, 0.3]);
        device.softmax_grad(samples, features, &mut x_grad, &out, &out_grad);
        assert_eq!(x_grad, Buffer::<f32>::from_slice(&[0.1, 0.2, 0.7, 0.3, 0.4, 0.3]));*/
    }
}