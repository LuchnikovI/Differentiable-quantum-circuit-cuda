use num_complex;
use std::os::raw::c_char;

#[cfg(not(feature = "f64"))]
use std::os::raw::c_float;

#[cfg(feature = "f64")]
use std::os::raw::c_double;

#[cfg(not(feature = "f64"))]
type Complex = num_complex::Complex<c_float>;
#[cfg(feature = "f64")]
type Complex = num_complex::Complex<c_double>;

extern "C" {
  pub(super) fn set2standard(
    state: *mut Complex,
    qubits_number: usize
  );
  pub(super) fn get_state(
    state: *mut *mut Complex,
    qubits_number: usize
  ) -> *const c_char;
  pub(super) fn drop_state(state: *mut Complex) -> *const c_char;
  pub(super) fn copy_to_host(
    state: *const Complex,
    host_state: *mut Complex,
    qubits_number: usize
  ) -> *const c_char;
  pub(super) fn q1gate(
    state: *mut Complex,
    gate: *const Complex,
    pos: usize,
    qubits_number: usize,
  ) -> *const c_char;
  pub(super) fn q1gate_inv(
    state: *mut Complex,
    gate: *const Complex,
    pos: usize,
    qubits_number: usize,
  ) -> *const c_char;
  pub(super) fn q2gate(
    state: *mut Complex,
    gate: *const Complex,
    pos2: usize,
    pos1: usize,
    qubits_number: usize,
  ) -> *const c_char;
  pub(super) fn q2gate_inv(
    state: *mut Complex,
    gate: *const Complex,
    pos2: usize,
    pos1: usize,
    qubits_number: usize,
  ) -> *const c_char;
  pub(super) fn q2gate_diag(
    state: *mut Complex,
    gate: *const Complex,
    pos2: usize,
    pos1: usize,
    qubits_number: usize,
  ) -> *const c_char;
  pub(super) fn set_from_host (
    device_state: *mut Complex,
    host_state: *const Complex,
    qubits_number: usize,
  ) -> *const c_char;
  pub(super) fn get_q1density(
    state: *const Complex,
    density: *mut Complex,
    pos: usize,
    qubits_number: usize,
  ) -> *const c_char;
  pub(super) fn get_q2density(
    state: *const Complex,
    density: *mut Complex,
    pos2: usize,
    pos1: usize,
    qubits_number: usize,
  ) -> *const c_char;
  pub(super) fn q1grad (
    fwd: *const Complex,
    bwd: *const Complex,
    grad: *mut Complex,
    pos: usize,
    qubits_number: usize,
  ) -> *const c_char;
  pub(super) fn q2grad (
    fwd: *const Complex,
    bwd: *const Complex,
    grad: *mut Complex,
    pos2: usize,
    pos1: usize,
    qubits_number: usize,
  ) -> *const c_char;
  pub(super) fn q2grad_diag (
    fwd: *const Complex,
    bwd: *const Complex,
    grad: *mut Complex,
    pos2: usize,
    pos1: usize,
    qubits_number: usize,
  ) -> *const c_char;
  pub(super) fn conj_and_double(
    src: *const Complex,
    dst: *mut Complex,
    qubits_number: usize,
  );
  pub(super) fn add(
    src: *const Complex,
    dst: *mut Complex,
    qubits_number: usize,
  );
  pub(super) fn copy(
    src: *const Complex,
    dst: *mut Complex,
    qubits_number: usize,
  );
}