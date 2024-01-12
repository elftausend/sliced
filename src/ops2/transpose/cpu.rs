use std::ops::Deref;

use custos::{impl_stack, Buffer, Device, Retrieve, Retriever, Shape, CPU};

use crate::{
    assign_or_set::{AssignOrSet, Set},
    Transpose,
};

pub fn slice_transpose<T: Clone, AOS: AssignOrSet<T>>(
    rows: usize,
    cols: usize,
    a: &[T],
    b: &mut [T],
) {
    for i in 0..rows {
        let index = i * cols;
        let row = &a[index..index + cols];

        for (index, row) in row.iter().enumerate() {
            let idx = rows * index + i;
            AOS::assign_or_set(&mut b[idx], row.clone());
            b[idx] = row.clone();
        }
    }
}

#[cfg(feature = "stack")]
use custos::Stack;

#[impl_stack]
impl<T, IS, OS, D, Mods: Retrieve<Self, T, OS>> Transpose<T, IS, OS, D> for CPU<Mods>
where
    T: Copy + Default,
    IS: Shape,
    OS: Shape,
    D: Device,
    D::Base<T, IS>: Deref<Target = [T]>,
{
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve(x.len(), x);
        slice_transpose::<_, Set>(rows, cols, x, &mut out);
        out
    }
}

/*impl<T: Clone, const ROWS: usize, const COLS: usize, IS: MayDim2<ROWS, COLS>, OS: MayDim2<COLS, ROWS>, D: MainMemory>
    Transpose2<T, ROWS, COLS, IS, OS, D> for CPU
{
    fn transpose(
        &self,
        rows: usize,
        cols: usize,
        x: &Buffer<T, D, IS>,
    ) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve(x.len());
        slice_transpose(rows, cols, &x, &mut out);
        out
    }
}*/

#[cfg(test)]
mod tests {
    use custos::{Buffer, CPU};

    use crate::Transpose;

    #[test]
    fn test_transpose() {
        let device = CPU::<custos::Base>::new();

        // 2 x 3
        let x = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        let out: Buffer<i32, CPU> = device.transpose(2, 3, &x);
        assert_eq!(&**out, [1, 4, 2, 5, 3, 6]);
    }
}
