use custos::{Buffer, Device, MainMemory, MayDim2, Shape, Transposed, CPU};

use crate::{Transpose, Transpose2};

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

impl<T: Clone, IS: Shape, OS: Shape, D: MainMemory> Transpose<T, IS, OS, D> for CPU {
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve(x.len());
        slice_transpose(rows, cols, &x, &mut out);
        out
    }
}

impl<T: Clone, const ROWS: usize, const COLS: usize, IS: MayDim2<ROWS, COLS>, D: MainMemory>
    Transpose2<T, ROWS, COLS, IS, D> for CPU
{
    fn transpose(
        &self,
        rows: usize,
        cols: usize,
        x: &Buffer<T, D, IS>,
    ) -> Buffer<T, Self, Transposed<ROWS, COLS, IS>> {
        let mut out: Buffer<T, CPU, Transposed<ROWS, COLS, IS>> = self.retrieve(x.len());
        slice_transpose(rows, cols, &x, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use custos::{Buffer, Transposed, CPU};

    use crate::Transpose2;

    #[test]
    fn test_transpose() {
        let device = CPU::new();

        let x = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        let out: Buffer<i32, CPU, Transposed> = device.transpose(2, 3, &x);
    }
}
