use custos::{static_api::StaticDevice, Alloc};

use crate::Matrix;

impl<'a, T: Clone> Matrix<'a, T> {
    /// Moves the `Matrix` to the static device of type `D`.
    /// # Example
    #[cfg_attr(not(feature = "opencl"), doc = "```ignore")]
    #[cfg_attr(feature = "opencl", doc = "```")]
    /// use sliced::Matrix;
    /// use custos::prelude::*;
    ///
    /// let a = Matrix::from((2, 3, [1, 2, 3, 4, 5, 6])).to_dev::<OpenCL>();
    /// assert_eq!(a.read(), &[1, 2, 3, 4, 5, 6])
    ///
    /// ```
    pub fn to_dev<D: StaticDevice + Alloc<'static, T> + 'static>(self) -> Matrix<'static, T, D> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.to_buf().to_dev(),
        }
    }
}
