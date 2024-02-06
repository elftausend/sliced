use std::{fmt::Display, ops::Deref};

use crate::{BinaryElementWise, Diagflat, Gemm, SoftmaxGrad, Transpose};
use custos::{
    prelude::Number, Base, Buffer, Device, GenericBlas, OnDropBuffer, Retrieve, Shape, CPU,
};

impl<T, S, Mods: OnDropBuffer + Retrieve<Self, T, ()>> SoftmaxGrad<T, S> for CPU<Mods>
where
    T: GenericBlas + Number,
    S: Shape,
    CPU: Gemm<T>, // for shape annotations
{
    fn softmax_grad(
        &self,
        samples: usize,
        features: usize,
        x_grad: &mut Buffer<T, Self, S>,
        out: &Buffer<T, Self, S>,
        out_grad: &Buffer<T, Self, S>,
    ) {
        let base_cpu = CPU::<Base>::new();

        for idx in 0..samples {
            let index = idx * features;

            // ensure that data is only read
            let single_out: Buffer<T> = unsafe {
                Buffer::from_raw_host((&out[index..index + features]).as_ptr() as *mut T, features)
            };

            // ensure that data is only read
            let single_grad: Buffer<T> = unsafe {
                Buffer::from_raw_host(
                    (&out_grad[index..index + features]).as_ptr() as *mut T,
                    features,
                )
            };

            let diagflat: Buffer<T, _> = base_cpu.diagflat(&single_out);

            // cols 1 x 1 cols
            let jacobian_matrix: Buffer<T, _> = base_cpu.sub(
                &diagflat,
                &base_cpu.gemm(
                    features,
                    1,
                    features,
                    &single_out,
                    &base_cpu.transpose(features, 1, &single_out),
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
    use crate::{Softmax, SoftmaxGrad};
    use custos::{Buffer, CPU};

    #[test]
    fn test_softmax_grad() {
        let device = CPU::<custos::Autograd<custos::Base>>::new();
        let x = Buffer::from((&device, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let out = device.softmax(2, 3, &x);
        crate::test_utils::roughly_equals(
            &*out,
            &[
                0.09003057, 0.24472847, 0.66524096, 0.09003057, 0.24472847, 0.66524096,
            ],
        );

        let mut x_grad = Buffer::from((&device, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        let out_grad = Buffer::from((&device, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        device.softmax_grad(2, 3, &mut x_grad, &out, &out_grad);
        crate::test_utils::roughly_equals(
            &*x_grad,
            &[
                -0.141871,
                -0.14077032,
                0.28258747,
                -0.141871,
                -0.14077032,
                0.28258747,
            ],
        );
    }

    #[cfg(feature = "matrix")]
    #[test]
    fn test_matrix_forward_softmax_grad() {
        use crate::{Matrix, MaxColsMayGrad};

        let device = CPU::<custos::Autograd<custos::Base>>::new();

        let x = Matrix::from((&device, 2, 3, [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]));

        // x.sub_cols(x.max_cols());
    }
}
