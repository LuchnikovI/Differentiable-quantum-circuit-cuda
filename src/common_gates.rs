use num_complex::Complex32;

pub fn get_hadamard() -> [Complex32; 4] {
  [
    Complex32::new(1. / f32::sqrt(2.), 0.), Complex32::new(1. / f32::sqrt(2.), 0.),
    Complex32::new(1. / f32::sqrt(2.), 0.), Complex32::new(-1. / f32::sqrt(2.), 0.),
  ]
}

pub fn get_cnot() -> [Complex32; 16] {
  [
    Complex32::new(1., 0.), Complex32::new(0., 0.), Complex32::new(0., 0.), Complex32::new(0., 0.),
    Complex32::new(0., 0.), Complex32::new(1., 0.), Complex32::new(0., 0.), Complex32::new(0., 0.),
    Complex32::new(0., 0.), Complex32::new(0., 0.), Complex32::new(0., 0.), Complex32::new(1., 0.),
    Complex32::new(0., 0.), Complex32::new(0., 0.), Complex32::new(1., 0.), Complex32::new(0., 0.),
  ]
}