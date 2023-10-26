use std::ops::{AddAssign, Mul, Deref};

use custos::{Buffer, Eval, Device, Resolve, Shape, ToVal, CPU};

use crate::ColOpGrad;

impl<T, LS, RS, D> ColOpGrad<T, LS, RS, D> for CPU
where
    T: Copy + AddAssign + Mul<Output = T>,
    LS: Shape,
    RS: Shape,
    D: Device,
    D::Data<T, LS>: Deref<Target = [T]>,
    D::Data<T, RS>: Deref<Target = [T]>
{
    #[inline]
    fn row_op_grad<LhsGrad, RhsGrad>(
        &self,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
        lhs_grad: &mut Buffer<T, D, LS>,
        rhs_grad: &mut Buffer<T, D, RS>,
        out_grad: &Buffer<T, D, LS>,
        lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LhsGrad,
        rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RhsGrad,
    ) where
        LhsGrad: Eval<T> + ToString,
        RhsGrad: Eval<T> + ToString,
    {
        slice_col_op_grad_lhs(cols, lhs, rhs, lhs_grad, out_grad, lhs_grad_fn);
        slice_col_op_grad_rhs(cols, lhs, rhs, rhs_grad, out_grad, rhs_grad_fn)
    }
}

// TODO: Fix row_op_grad, row op only supports rhs for grad fn
pub fn slice_col_op_grad_lhs<T, GRAD>(
    cols: usize,
    lhs: &[T],
    rhs: &[T],
    lhs_grad: &mut [T],
    out_grad: &[T],
    lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> GRAD,
) where
    GRAD: Eval<T>,
    T: Copy + AddAssign + Mul<Output = T>,
{
    for (((lhs, lhs_grad), out_grad), rhs) in lhs
        .chunks(cols)
        .zip(lhs_grad.chunks_mut(cols))
        .zip(out_grad.chunks(cols))
        .zip(rhs)
    {
        for ((lhs, lhs_grad), out_grad) in lhs.iter().zip(lhs_grad).zip(out_grad) {
            *lhs_grad += lhs_grad_fn((*lhs).to_val(), (*rhs).to_val()).eval() * *out_grad;
        }
    }
}

// combine with lhs grad version
pub fn slice_col_op_grad_rhs<T, RhsGrad>(
    cols: usize,
    lhs: &[T],
    rhs: &[T],
    rhs_grad: &mut [T],
    out_grad: &[T],
    rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RhsGrad,
) where
    RhsGrad: Eval<T>,
    T: Copy + AddAssign + Mul<Output = T>,
{
    for (((lhs, out_grad), rhs), rhs_grad) in lhs
        .chunks(cols)
        .zip(out_grad.chunks(cols))
        .zip(rhs)
        .zip(rhs_grad)
    {
        for (lhs, out_grad) in lhs.iter().zip(out_grad) {
            *rhs_grad += rhs_grad_fn((*lhs).to_val(), (*rhs).to_val()).eval() * *out_grad;
        }
    }
    /*let mut rhs_idx = 0;
    for (idx, (lhs, out_grad)) in lhs.iter().zip(out_grad).enumerate() {
        if (idx + 1) % (cols + 1) == 0 {
            rhs_idx += 1;
        }
        rhs_grad[rhs_idx] += rhs_grad_fn((*lhs).to_val(), rhs[rhs_idx].to_val()).eval() * *out_grad;
    }*/
}

#[cfg(test)]
mod tests {
    use custos::{Combiner, ToVal};

    use crate::{slice_col_op_grad_lhs, slice_col_op_grad_rhs, test_utils::roughly_equals};

    #[test]
    fn test_slice_col_op_grad_lhs_div() {
        #[rustfmt::skip]
        let lhs = [
            1., 2., 3., 
            4., 5., 6.
        ];

        let mut lhs_grad = [0.; 6];

        let rhs = [-3., 2.];

        slice_col_op_grad_lhs(
            3,
            &lhs,
            &rhs,
            &mut lhs_grad,
            &[1.4, 2.5, 3.3, 4., 5., 6.],
            |_, rhs| (1f32).to_val().div(rhs),
        );

        roughly_equals(
            &lhs_grad,
            &[
                1.4 * 1. / -3.,
                2.5 * 1. / -3.,
                3.3 * 1. / -3.,
                4. * 1. / 2.,
                5. * 1. / 2.,
                6. * 1. / 2.,
            ],
        );
    }

    #[test]
    fn test_slice_col_op_grad_rhs_div() {
        #[rustfmt::skip]
        let lhs = [
            1., 2., 3., 
            4., 5., 6.
        ];

        let rhs = [-3., 2.];

        let mut rhs_grad = [0.; 2];

        let gf = |lhs: f64, rhs: f64| lhs / -(rhs * rhs);

        slice_col_op_grad_rhs(
            3,
            &lhs,
            &rhs,
            &mut rhs_grad,
            &[1.4, 2.5, 3.3, 4., 5., 6.],
            |lhs, rhs| lhs.div(rhs.mul(rhs).neg()),
        );

        roughly_equals(
            &rhs_grad,
            &[
                gf(1., -3.) * 1.4 + gf(2., -3.) * 2.5 + gf(3., -3.) * 3.3,
                gf(4., 2.) * 4. + gf(5., 2.) * 5. + gf(6., 2.) * 6.,
            ],
        );
    }
}
