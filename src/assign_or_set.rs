use std::ops::AddAssign;

/// Either `+=` or `=`.
///
/// # Example
/// ```
/// use sliced::assign_or_set::{Assign, AssignOrSet, Set};
///
/// let mut x = 3;
/// Set::assign_or_set(&mut x, 4);
/// assert_eq!(x, 4);
///
/// Assign::assign_or_set(&mut x, 4);
/// assert_eq!(x, 8);
///
/// // useful for OpenCL kernels:
///
/// fn aos<T, AOS: AssignOrSet<T>>() -> &'static str {
///     AOS::STR_OP
/// }
/// assert_eq!(aos::<u8, Assign>(), "+=");
/// assert_eq!(aos::<u8, Set>(), "=")
/// ```
pub trait AssignOrSet<T> {
    const STR_OP: &'static str;
    fn assign_or_set(lhs: &mut T, rhs: T);
}

pub struct Assign;

impl<T: AddAssign> AssignOrSet<T> for Assign {
    const STR_OP: &'static str = "+=";
    #[inline]
    fn assign_or_set(lhs: &mut T, rhs: T) {
        *lhs += rhs;
    }
}

pub struct Set;

impl<T> AssignOrSet<T> for Set {
    const STR_OP: &'static str = "=";
    #[inline]
    fn assign_or_set(lhs: &mut T, rhs: T) {
        *lhs = rhs;
    }
}
