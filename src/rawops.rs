use custos::{Buffer, Device, Eval, Resolve, Shape};

mod cpu;

#[cfg(feature = "opencl")]
mod opencl;

pub trait BinaryElementWise<T, S: Shape = (), D: Device = Self>: Device {
    fn binary_ew<O>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        f: impl Fn(Resolve<T>, Resolve<T>) -> O,
    ) -> Buffer<T, Self, S>
    where
        O: Eval<T> + ToString;
}

pub trait BinaryGrad<T, S: Shape = (), D: Device = Self>: Device {
    fn add_binary_grad<LO, RO>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        lhs_grad: &mut Buffer<T, D, S>,
        rhs_grad: &mut Buffer<T, D, S>,
        out: &Buffer<T, D, S>,
        lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LO,
        rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RO,
    ) where
        LO: Eval<T> + ToString,
        RO: Eval<T> + ToString;
}

pub trait BinaryEWWithGrad<T, S: Shape = (), D: Device = Self>: Device {
    fn binary_ew_w_grad<FO, LO, RO>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        forward_fn: impl Fn(Resolve<T>, Resolve<T>) -> FO,
        lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LO,
        rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RO,
    ) -> Buffer<T, Self, S>
    where
        FO: Eval<T> + ToString,
        LO: Eval<T> + ToString,
        RO: Eval<T> + ToString;
}
