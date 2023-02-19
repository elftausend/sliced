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
