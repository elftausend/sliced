use std::ops::AddAssign;

use custos::{Buffer, Device, CPU};

use crate::Diagflat;

impl<T: Copy> Diagflat<T> for CPU {
    fn diagflat(&self, x: &Buffer<T>) -> Buffer<T> {
        let mut out = self.retrieve(x.len() * x.len());
        diagflat(x, &mut out);
        out
    }
}

/// Takes the values of slice `x` and puts it diagonally on the slice `out`.
///
/// # Example
/// ```
/// use sliced::diagflat;
///
/// let x = [2, 1, 3, -3, 3];
///
/// let mut out = [
///     0, 0, 0, 0, 0,
///     0, 0, 0, 0, 0,
///     0, 0, 0, 0, 0,
///     0, 0, 0, 0, 0,
///     0, 0, 0, 0, 0,
/// ];
///
/// diagflat(&x, &mut out);
///
/// let actual_out = [
///     2, 0, 0, 0, 0,
///     0, 1, 0, 0, 0,
///     0, 0, 3, 0, 0,
///     0, 0, 0, -3, 0,
///     0, 0, 0, 0, 3,
/// ];
///
/// assert_eq!(out, actual_out)
/// ```
///
pub fn diagflat<T: Copy>(x: &[T], out: &mut [T]) {
    for (idx, val) in x.iter().enumerate() {
        out[x.len() * idx + idx] = *val;
    }
}

/// Computes the gradient for the diagflat operation.
///
/// # Example
/// ```
/// use sliced::diagflat_grad;
///
/// let mut x_grad = [0; 4];
///
/// let out_grad = [
///     4, 1, 3, -3,
///     2, 5, 3, 1,
///     2, 6, 3, -3,
///     2, 7, 3, -3
/// ];
///
/// diagflat_grad(&mut x_grad, &out_grad);
///
/// assert_eq!(x_grad, [4, 5, 3, -3])
/// ```
pub fn diagflat_grad<T: Copy + AddAssign>(x_grad: &mut [T], out_grad: &[T]) {
    for idx in 0..x_grad.len() {
        x_grad[idx] += out_grad[x_grad.len() * idx + idx];
    }
}

#[cfg(test)]
mod tests {
    use crate::{diagflat, diagflat_grad};

    #[test]
    fn test_diagflat() {
        let x = [2, 1, 3, -3, 3];

        #[rustfmt::skip]
        let mut out = [
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ];

        diagflat(&x, &mut out);

        #[rustfmt::skip]
        let actual_out = [
            2, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 3, 0, 0,
            0, 0, 0, -3, 0,
            0, 0, 0, 0, 3,
        ];

        assert_eq!(out, actual_out)
    }

    #[test]
    fn test_diagflat_grad() {
        let mut x_grad = [0; 4];

        #[rustfmt::skip]
        let out_grad = [
            4, 1, 3, -3, 
            2, 5, 3, 1, 
            2, 6, 3, -3,
            2, 7, 3, -3
        ];

        diagflat_grad(&mut x_grad, &out_grad);

        assert_eq!(x_grad, [4, 5, 3, -3])
    }
}
