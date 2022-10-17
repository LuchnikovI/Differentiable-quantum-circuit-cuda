use num_complex;

#[cfg(not(feature = "f64"))]
use std::os::raw::c_float;

#[cfg(feature = "f64")]
use std::os::raw::c_double;

#[cfg(not(feature = "f64"))]
type Complex = num_complex::Complex<c_float>;
#[cfg(feature = "f64")]
type Complex = num_complex::Complex<c_double>;
#[cfg(not(feature = "f64"))]
type Float = c_float;
#[cfg(feature = "f64")]
type Float = c_double;

#[allow(dead_code)]
pub fn get_hadamard() -> [Complex; 4] {
  [
    Complex::new(1. / Float::sqrt(2.), 0.), Complex::new(1. / Float::sqrt(2.), 0.),
    Complex::new(1. / Float::sqrt(2.), 0.), Complex::new(-1. / Float::sqrt(2.), 0.),
  ]
}

#[allow(dead_code)]
pub fn get_cnot() -> [Complex; 16] {
  [
    Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
    Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
    Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.),
    Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.),
  ]
}