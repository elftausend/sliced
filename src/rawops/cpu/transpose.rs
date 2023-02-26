use custos::{Buffer, Device, MainMemory, Shape, CPU, impl_stack};

use crate::Transpose;

pub fn slice_transpose<T: Clone>(rows: usize, cols: usize, a: &[T], b: &mut [T]) {
    for i in 0..rows {
        let index = i * cols;
        let row = &a[index..index + cols];

        for (index, row) in row.iter().enumerate() {
            let idx = rows * index + i;
            b[idx] = row.clone();
        }
    }
}

#[cfg(feature = "stack")]
use custos::Stack;

#[impl_stack]
impl<T: Copy + Default, IS: Shape, OS: Shape, D: MainMemory> Transpose<T, IS, OS, D> for CPU {
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve(x.len());
        slice_transpose(rows, cols, &x, &mut out);
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
        let device = CPU::new();

        // 2 x 3
        let x = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        let out: Buffer<i32, CPU> = device.transpose(2, 3, &x);
        assert_eq!(&*out, [1, 4, 2, 5, 3, 6]);
    }
}
