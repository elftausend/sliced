#[cfg(any(feature = "cpu", feature = "stack"))]
#[cfg(feature = "autograd")]
mod cpu_stack;

#[cfg(any(feature = "cpu", feature = "stack"))]
#[cfg(feature = "autograd")]
pub use cpu_stack::*;

#[cfg(feature = "opencl")]
#[cfg(feature = "autograd")]
mod opencl;

#[cfg(feature = "opencl")]
#[cfg(feature = "autograd")]
pub use opencl::*;

use custos::{using_autograd, Buffer, Device, Shape};

#[cfg(feature = "autograd")]
use custos::{Eval, MayToCLSource, Resolve};

#[using_autograd]
pub trait BinaryElementWiseGrad<T, S: Shape = (), D: Device = Self>: Device {
    #[track_caller]
    fn binary_ew_grad<LO, RO>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        lhs_grad: &mut Buffer<T, D, S>,
        rhs_grad: &mut Buffer<T, D, S>,
        out: &Buffer<T, D, S>,
        lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LO,
        rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RO,
    ) where
        LO: Eval<T> + MayToCLSource,
        RO: Eval<T> + MayToCLSource;
}

pub trait AddElementWiseGrad<T, S: Shape = (), D: Device = Self>: Device {
    #[track_caller]
    fn add_ew_grad(
        &self,
        lhs_grad: &mut Buffer<T, D, S>,
        rhs_grad: &mut Buffer<T, D, S>,
        out: &Buffer<T, D, S>,
    );
}
