#[cfg(feature = "matrix")]
mod matrix;
mod ops;
mod rawops;

pub use ops::*;
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
}
