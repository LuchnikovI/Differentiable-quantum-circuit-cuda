use num_complex;
use std::os::raw::{
  c_double,
  c_float,
};
use std::iter::zip;

#[cfg(not(feature = "f64"))]
type Complex = num_complex::Complex<c_float>;
#[cfg(feature = "f64")]
type Complex = num_complex::Complex<c_double>;
#[cfg(not(feature = "f64"))]
type Float = c_float;
#[cfg(feature = "f64")]
type Float = c_double;

pub(super) fn cmp_complex_slices<'a>(
  slice1: impl IntoIterator<Item = &'a Complex>,
  slice2: impl IntoIterator<Item = &'a Complex>,
  tol: Float,
)
{
  zip(slice1.into_iter(), slice2.into_iter())
    .enumerate()
    .for_each(|(idx, (lhs, rhs))| {
      let diff = lhs - rhs;
      let max_abs = Float::max(
        lhs.re * lhs.re + lhs.im * lhs.im,
        rhs.re * rhs.re + rhs.im * rhs.im,
      ).sqrt();
      if max_abs != 0. {
        let flag = (diff.re * diff.re + diff.im * diff.im).sqrt() / max_abs < tol;
        if !flag {
          panic!("Elements number {} are too different: lhs: {}, rhs: {}", idx, lhs, rhs)
        }
      }
    });
}