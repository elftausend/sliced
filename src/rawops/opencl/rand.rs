use custos::{
    prelude::{cpu_exec_unary_may_unified_mut, Float},
    OpenCL, Buffer,
};

use crate::RandOp;

impl<T: Float> RandOp<T> for OpenCL {
    #[inline]
    fn rand(&self, x: &mut Buffer<T, Self>, lo: T, hi: T) {
        cpu_exec_unary_may_unified_mut(self, x, |cpu, x| cpu.rand(x, lo, hi)).unwrap();
    }
}
