use custos::{
    opencl::CLBuffer,
    prelude::{cpu_exec_unary_may_unified_mut, Float},
    OpenCL,
};

use crate::RandOp;

impl<T: Float> RandOp<T> for OpenCL {
    #[inline]
    fn rand(&self, x: &mut CLBuffer<T>, lo: T, hi: T) {
        cpu_exec_unary_may_unified_mut(self, x, |cpu, x| cpu.rand(x, lo, hi)).unwrap();
    }
}
