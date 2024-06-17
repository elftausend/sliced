mod grad;
pub use grad::*;

#[cfg(any(feature = "cpu", feature = "stack"))]
mod cpu_stack;

#[cfg(any(feature = "cpu", feature = "stack"))]
pub use cpu_stack::*;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(feature = "opencl")]
pub use opencl::*;

use core::fmt::Display;

use custos::{impl_nnapi_op, Buffer, Combiner, Device, Eval, MayToCLSource, Resolve, Shape};

#[cfg(feature = "nnapi")]
use custos::nnapi::{nnapi_sys::OperationCode, Operand};

// pub trait BinaryElementWise2<T, S: Shape = (), D: Device = Self>: Device {
//     fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>;
// }

// #[cfg(feature = "nnapi")]
// impl<T: custos::nnapi::AsOperandCode, S: Shape> BinaryElementWise2<T, S> for custos::NnapiDevice {
//     fn add(&self, lhs: &Buffer<T, Self, S>, rhs: &Buffer<T, Self, S>) -> Buffer<T, Self, S> {
//         self.retrieve_with_init::<T, S>(S::LEN, |out| {
//             let activation_idx = self.add_operand(&Operand::activation()).unwrap();
//             let mut model = self.model.borrow_mut();

//             model
//                 .set_activation_operand_value(activation_idx as i32)
//                 .unwrap();
//             model
//                 .add_operation(
//                     OperationCode::ANEURALNETWORKS_ADD,
//                     &[lhs.ptr.idx, rhs.ptr.idx, activation_idx],
//                     &[out.ptr.idx],
//                 )
//                 .unwrap();
//         })
//     }
// }

#[impl_nnapi_op(
    None,
    ANEURALNETWORKS_ADD,
    ANEURALNETWORKS_MUL,
    ANEURALNETWORKS_DIV,
    ANEURALNETWORKS_SUB
)]
pub trait BinaryElementWise<T: Copy + 'static, S: Shape = (), D: Device = Self>: Device {
    fn binary_ew<O>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        f: impl Fn(Resolve<T>, Resolve<T>) -> O + Copy + 'static,
    ) -> Buffer<T, Self, S>
    where
        O: Eval<T> + MayToCLSource;

    #[inline]

    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>
    where
        T: core::ops::Add<T, Output = T>,
    {
        self.binary_ew(lhs, rhs, |lhs, rhs| lhs.add(rhs))
    }

    #[inline]

    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>
    where
        T: core::ops::Mul<T, Output = T>,
    {
        self.binary_ew(lhs, rhs, |lhs, rhs| lhs.mul(rhs))
    }

    #[inline]

    fn div(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>
    where
        T: core::ops::Div<T, Output = T>,
    {
        self.binary_ew(lhs, rhs, |lhs, rhs| lhs.div(rhs))
    }

    #[inline]

    fn sub(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>
    where
        T: core::ops::Sub<T, Output = T>,
    {
        self.binary_ew(lhs, rhs, |lhs, rhs| lhs.sub(rhs))
    }
}
