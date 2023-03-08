use std::ops::AddAssign;
use custos::{Buffer, Shape, CPU};
use crate::DiagflatGrad;


// TODO stack impl
impl<T: Copy + AddAssign, IS: Shape, OS: Shape> DiagflatGrad<T, IS, OS> for CPU {
    #[inline]
    fn diagflat_grad(&self, x_grad: &mut Buffer<T, Self, IS>, out_grad: &Buffer<T, Self, OS>) {
        diagflat_grad(x_grad, out_grad);
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
    use crate::diagflat_grad;

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
