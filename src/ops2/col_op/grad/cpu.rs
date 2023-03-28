use std::ops::{AddAssign, Mul};

pub fn slice_col_op_grad_lhs<T>(
    cols: usize,
    lhs: &[T],
    rhs: &[T],
    lhs_grad: &mut [T],
    rhs_grad: &mut [T],
    out_grad: &[T],
    lhs_grad_fn: impl Fn(T, T) -> T,
    rhs_grad_fn: impl Fn(T) -> T,
) where
    T: Copy + AddAssign + Mul<Output = T>,
{
    let mut rhs_cycle = rhs.iter().cycle();
    for ((lhs, lhs_grad), out_grad) in lhs.iter().zip(lhs_grad).zip(out_grad) {
        let rhs = rhs_cycle.next().unwrap();
        *lhs_grad = lhs_grad_fn(*lhs, *rhs) * *out_grad;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_slice_col_op_grad_lhs_div() {

    }
}