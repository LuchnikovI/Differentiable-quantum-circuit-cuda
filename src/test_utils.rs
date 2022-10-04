use num_complex::Complex32;
use std::iter::zip;

pub(super) fn cmp_complex_slices<'a>(
  slice1: impl IntoIterator<Item = &'a Complex32>,
  slice2: impl IntoIterator<Item = &'a Complex32>,
  tol: f32,
)
{
  zip(slice1.into_iter(), slice2.into_iter())
    .enumerate()
    .for_each(|(idx, (lhs, rhs))| {
      let diff = lhs - rhs;
      let max_abs = f32::max(
        lhs.re * lhs.re + lhs.im * lhs.im,
        rhs.re * rhs.re + rhs.im * rhs.im,
      ).sqrt();
      if max_abs != 0. {
        let flag = (diff.re * diff.re + diff.im * diff.im).sqrt() / max_abs < tol;
        if !flag {
          println!("Elements number {} are too different: lhs: {}, rhs: {}", idx, lhs, rhs)
        }
      }
    });
}