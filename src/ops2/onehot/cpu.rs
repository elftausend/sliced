use custos::{prelude::Number, Buffer, Device, Retrieve, Retriever, CPU};

use crate::{max, Onehot};

impl<T: PartialOrd + Number, Mods: Retrieve<Self, T>> Onehot<T> for CPU<Mods> {
    #[inline]
    fn onehot(&self, classes: &Buffer<T, Self>) -> Buffer<T, Self> {
        let highest_class = max(classes).unwrap().as_usize() + 1;

        let mut out = self
            .retrieve(classes.len() * highest_class, classes)
            .unwrap();
        slice_onehot(highest_class, classes, &mut out);

        out
    }
}

/// Onehot encodes a slice of classes.
///
/// # Example
///
/// ```
/// use sliced::{max, slice_onehot};
///
/// let classes = [2, 3, 0, 1];
/// let highest_class = max(&classes).unwrap() + 1;
/// let mut onehot_classes = vec![0; classes.len() * highest_class];
///
/// slice_onehot(highest_class, &classes, &mut onehot_classes);
///
/// assert_eq!([
///     0, 0, 1, 0,
///     0, 0, 0, 1,
///     1, 0, 0, 0,
///     0, 1, 0, 0
/// ], &*onehot_classes);
///
/// ```
pub fn slice_onehot<T: Number>(highest_class: usize, classes: &[T], onehot: &mut [T]) {
    for (onehot_row, class) in classes.iter().enumerate() {
        onehot[onehot_row * highest_class + class.as_usize()] = T::one();
    }
}

#[cfg(test)]
mod tests {
    use crate::{max, slice_onehot};

    #[test]
    fn test_onehot() {
        // highest class: 6
        let classes = [0, 1, 4, 3, 0, 5, 2, 1];
        let highest_class = max(&classes).unwrap() + 1;
        let mut onehot_classes = vec![0; classes.len() * highest_class];

        slice_onehot(highest_class, &classes, &mut onehot_classes);

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
}
