mod grad;
use custos::{impl_nnapi_op, Buffer, Device, Shape};
pub use grad::*;

#[cfg(any(feature = "cpu", feature = "stack"))]
mod cpu_stack;

#[cfg(any(feature = "cpu", feature = "stack"))]
pub use cpu_stack::*;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(feature = "opencl")]
pub use opencl::*;

#[cfg(feature = "nnapi")]
use custos::nnapi::nnapi_sys::OperationCode;

#[impl_nnapi_op(ANEURALNETWORKS_BATCH_MATMUL)]
pub trait Gemm<T, LS: Shape = (), RS: Shape = (), OS: Shape = (), D: Device = Self>:
    Device
{
    #[track_caller]
    fn gemm(
        &self,
        m: usize,
        k: usize,
        n: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, Self, OS>;
}

#[cfg(test)]
mod test {

    #[cfg(feature = "nnapi")]
    #[test]
    fn test_gemm_nnapi() -> custos::Result<()> {
        use custos::{Buffer, Dim2, NnapiDevice, WithShape};

        use crate::Gemm;

        let device = NnapiDevice::new()?;

        let m = 4;
        let k = 2;
        let n = 3;

        // 4 x 2
        #[rustfmt::skip]
        let lhs = Buffer::with(&device, [
            [1., 2.,], 
            [3., 4.,],
            [4., 5.,], 
            [6., 5.,],
        ]);

        // 2 x 3
        let rhs = Buffer::with(&device, [[1., 2., 3.], [4., 5., 6.]]);

        let out: Buffer<_, _, Dim2<4, 3>> = device.gemm(m, k, n, &lhs, &rhs);

        Ok(())
    }
}
