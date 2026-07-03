#![cfg_attr(not(feature = "std"), no_std)]

pub mod assign_or_set;
#[cfg(feature = "matrix")]
mod matrix;
mod ops;
mod ops2;
mod rawops;

pub use ops::*;
pub use ops2::*;
pub use rawops::*;

#[cfg(feature = "matrix")]
pub use matrix::*;

#[cfg(feature = "cpu")]
pub use ::custos::CPU;

#[cfg(feature = "stack")]
pub use ::custos::Stack;

#[cfg(feature = "opencl")]
pub use ::custos::OpenCL;

#[cfg(feature = "cuda")]
pub use ::custos::CUDA;

pub use ::custos::Buffer;

pub mod custos {
    pub use custos::*;
}

pub mod test_utils {
    use custos::prelude::Float;

    pub fn roughly_equals<T: Float>(lhs: &[T], rhs: &[T]) {
        for (a, b) in lhs.iter().zip(rhs) {
            let abs = (*a - *b).abs();
            if abs > T::one() / T::from_u64(100) {
                panic!(
                    "\n left: '{:?}',\n right: '{:?}', \n left elem.: {} != right elem. {}",
                    lhs, rhs, a, b
                )
            }
        }
    }

    /// Reads a (deviceless) gradient buffer.
    /// `device.read(grad_buf)` alone is ambiguous for type inference, as `Read` is generic
    /// over the buffer's `Shape`; fixing `S = ()` here via the signature resolves it.
    pub fn read_grad<'a, T, D>(device: &'a D, buf: &'a custos::Buffer<'static, T, D>) -> D::Read<'a>
    where
        T: custos::Unit + Default + Clone,
        D: custos::Read<T>,
    {
        device.read(buf)
    }
}
