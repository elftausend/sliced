use custos::prelude::Number;

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
    use crate::{max, onehot_grad};

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
