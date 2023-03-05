/*copilot pub trait ColOpGrad<T, LS: Shape = (), RS: Shape = (), D: Device = Self>: Device {
    fn col_op_grad<F>(
        &self,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
        f: F,
    ) -> (Buffer<T, Self, LS>, Buffer<T, Self, RS>)
    where
        F: Fn(T, T) -> (T, T) + Copy;
}*/
