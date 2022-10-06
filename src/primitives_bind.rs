use num_complex::Complex32;

extern "C" {
  pub(super) fn set2standard(
    state: *mut Complex32,
    qubits_number: usize
  );
  pub(super) fn get_state(
    state: *mut *mut Complex32,
    qubits_number: usize
  ) -> i32;
  pub(super) fn drop_state(state: *mut Complex32) -> i32;
  pub(super) fn copy_to_host(
    state: *const Complex32,
    host_state: *mut Complex32,
    qubits_number: usize
  ) -> i32;
  pub(super) fn q1gate(
    state: *mut Complex32,
    gate: *const Complex32,
    pos: usize,
    qubits_number: usize,
  ) -> i32;
  pub(super) fn q2gate(
    state: *mut Complex32,
    gate: *const Complex32,
    pos2: usize,
    pos1: usize,
    qubits_number: usize,
  ) -> i32;
  pub(super) fn set_from_host (
    device_state: *mut Complex32,
    host_state: *const Complex32,
    qubits_number: usize,
  ) -> i32;
  pub(super) fn get_q1density(
    state: *const Complex32,
    density: *mut Complex32,
    pos: usize,
    qubits_number: usize,
  ) -> i32;
  pub(super) fn get_q2density(
    state: *const Complex32,
    density: *mut Complex32,
    pos2: usize,
    pos1: usize,
    qubits_number: usize,
  ) -> i32;
  pub(super) fn q1grad (
    fwd: *const Complex32,
    bwd: *const Complex32,
    grad: *mut Complex32,
    pos: usize,
    qubits_number: usize,
  ) -> i32;
  pub(super) fn q2grad (
    fwd: *const Complex32,
    bwd: *const Complex32,
    grad: *mut Complex32,
    pos2: usize,
    pos1: usize,
    qubits_number: usize,
  ) -> i32;
  pub(super) fn conj_and_double(
    src: *const Complex32,
    dst: *mut Complex32,
    qubits_number: usize,
  );
  pub(super) fn add(
    src: *const Complex32,
    dst: *mut Complex32,
    qubits_number: usize,
  );
  pub(super) fn copy(
    src: *const Complex32,
    dst: *mut Complex32,
    qubits_number: usize,
  );
}