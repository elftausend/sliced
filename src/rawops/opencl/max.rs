use custos::{OpenCL, Buffer};

use crate::Max;


impl<T> Max<T> for OpenCL {
    fn max(&self, x: &Buffer<T, Self>) -> T {
        todo!()
    }
}