use crate::{max, Onehot};
use custos::{prelude::Number, Buffer, Device, CPU};

impl<T: Ord + Number> Onehot<T> for CPU {
    #[inline]
    fn onehot(&self, classes: &Buffer<T>) -> Buffer<T> {
        let highest_class = max(classes).unwrap().as_usize() + 1;

        let mut out = self.retrieve(classes.len() * highest_class, classes);
        onehot(highest_class, classes, &mut out);

        out
    }
}

/// Onehot encodes a slice of classes.
///
/// # Example
///
/// ```
/// use sliced::{max, onehot};
///
/// let classes = [2, 3, 0, 1];
/// let highest_class = max(&classes).unwrap() + 1;
/// let mut onehot_classes = vec![0; classes.len() * highest_class];
///
/// onehot(highest_class, &classes, &mut onehot_classes);
///
/// assert_eq!([
///     0, 0, 1, 0,
///     0, 0, 0, 1,
///     1, 0, 0, 0,
///     0, 1, 0, 0
/// ], &*onehot_classes);
///
/// ```
pub fn onehot<T: Number>(highest_class: usize, classes: &[T], onehot: &mut [T]) {
    for (onehot_row, class) in classes.iter().enumerate() {
        onehot[onehot_row * highest_class + class.as_usize()] = T::one();
    }
}

pub fn onehot_grad<T: Number>(
    highest_class: usize,
    classes: &[T],
    classes_grad: &mut [T],
    out_grad: &[T],
) {
    for (onehot_row, class) in classes.iter().enumerate() {
        classes_grad[onehot_row] += out_grad[onehot_row * highest_class + class.as_usize()];
    }
}

#[cfg(test)]
mod tests {
    use super::onehot;
    use crate::{onehot_grad, rawops::cpu::max};

    #[test]
    fn test_onehot() {
        // highest class: 6
        let classes = [0, 1, 4, 3, 0, 5, 2, 1];
        let highest_class = max(&classes).unwrap() + 1;
        let mut onehot_classes = vec![0; classes.len() * highest_class];

        onehot(highest_class, &classes, &mut onehot_classes);

        #[rustfmt::skip]
        let cmp_to = 
        [
            1, 0, 0, 0, 0, 0, 
            0, 1, 0, 0, 0, 0, 
            0, 0, 0, 0, 1, 0, 
            0, 0, 0, 1, 0, 0, 
            1, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 1, 
            0, 0, 1, 0, 0, 0, 
            0, 1, 0, 0, 0, 0
        ];

        assert_eq!(cmp_to, &*onehot_classes)
    }

    #[test]
    fn test_onehot_grad() {
        let classes = [0, 1, 4, 3];
        let highest_class = max(&classes).unwrap() + 1;

        let mut classes_grad = vec![0; classes.len()];
        let out_grad = [
            2, -1, 3, 7, -2, 1, 3, 2, 4, 3, 3, -2, 3, -2, 1, 5, 6, -1, 4, 5,
        ];

        onehot_grad(
            highest_class as usize,
            &classes,
            &mut classes_grad,
            &out_grad,
        );

        let expected = [2, 3, 1, 4];

        assert_eq!(expected, &*classes_grad);
    }
}
