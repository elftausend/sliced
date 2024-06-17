use crate::Diagflat;
use custos::{Buffer, Device, Retriever, Shape, CPU};

// TODO stack impl
impl<T: Copy, IS: Shape, OS: Shape> Diagflat<T, IS, OS> for CPU {
    fn diagflat(&self, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve(x.len() * x.len(), x).unwrap();
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

#[cfg(test)]
mod tests {
    use crate::diagflat;

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
}
