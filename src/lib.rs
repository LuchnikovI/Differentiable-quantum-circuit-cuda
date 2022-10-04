mod primitives_bind;
mod quantized_tensor;
mod common_gates;
mod test_utils;

#[cfg(test)]
mod tests {
  use std::ptr;
  use num_complex::Complex32;
  use crate::primitives_bind::{
    get_state,
    drop_state,
    copy_to_host,
    set2standard,
  };
  #[test]
  fn test() {
    unsafe {
      let qubits_number = 5;
      let mut host_state = Vec::with_capacity(1 << qubits_number);
      host_state.set_len(1 << qubits_number);
      let mut state: *mut Complex32 = ptr::null_mut();
      println!("State addres before allocation:{:?}", state);
      let alloc_status = get_state(&mut state, qubits_number);
      println!("State addres after allocation:{:?}", state);
      println!("Allocations status: {}", alloc_status);
      set2standard(state, qubits_number);
      let copy_status = copy_to_host(state, host_state.as_mut_ptr(), qubits_number);
      println!("Copy status: {}", copy_status);
      println!("{:?}", host_state);
      let drop_status = drop_state(state);
      println!("Drop status: {}", drop_status);
    }
  }
}