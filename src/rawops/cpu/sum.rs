use std::ops::AddAssign;

pub fn sum_rows<T: AddAssign + Copy>(rows: usize, cols: usize, x: &[T], out: &mut [T]) {
    for idx in 0..rows {
        let index = idx * cols;
        let row = &x[index..index + cols];

        for (i, value) in row.iter().enumerate() {
            out[i] += *value;
        }
    }
}
