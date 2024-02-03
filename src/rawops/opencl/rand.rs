use custos::{
    prelude::{cpu_exec_unary_may_unified_mut, Float},
    Buffer, OnDropBuffer, OpenCL,
};

use crate::RandOp;

impl<T: Float, Mods: OnDropBuffer + 'static> RandOp<T> for OpenCL<Mods> {
    #[inline]
    fn rand(&self, x: &mut Buffer<T, Self>, lo: T, hi: T) {
        cpu_exec_unary_may_unified_mut(self, x, |cpu, x| cpu.rand(x, lo, hi)).unwrap();
    }
}
