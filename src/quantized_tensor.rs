use num_complex::Complex32;
use super::primitives_bind::{
  copy_to_host,
  set2standard,
  get_state,
  drop_state,
  q1gate,
  q2gate,
  set_from_host,
  get_q1density,
  get_q2density,
  q1grad,
  q2grad,
  conj_and_double,
  add,
  copy,
};

pub struct QuantizedTensor {
  state_ptr: *mut Complex32,
  qubits_number: usize,
}

impl QuantizedTensor {
  pub fn new_standard(qubits_number: usize) -> Self {
    let mut state_ptr = std::ptr::null_mut();
    let cuda_status = unsafe { get_state(&mut state_ptr, qubits_number) };
    if cuda_status == 0
    {
      unsafe { set2standard(state_ptr, qubits_number) };
      Self { state_ptr, qubits_number }
    } else { panic!("Cuda error with code {} occurred during state allocation.", cuda_status)}
  }
  pub fn new_from_host(state: &[Complex32]) -> Self {
    let mut size = state.len();
    assert_eq!(size & (size - 1), 0, "State size is not a power of 2.");
    let qubits_number = {
      let mut counter: usize = 0;
      while size != 1 {
        size = size >> 1;
        counter += 1;
      }
      counter
    };
    let mut state_ptr = std::ptr::null_mut();
    let mut cuda_status = unsafe { get_state(&mut state_ptr, qubits_number) };
    if cuda_status == 0
    {
      unsafe {
        cuda_status = set_from_host(state_ptr, state.as_ptr(), qubits_number);
        if cuda_status != 0 { panic!("Cuda error with code {} occurred during copying from CPU to GPU.", cuda_status) }
      };
      Self { state_ptr, qubits_number }
    } else { panic!("Cuda error with code {} occurred during state allocation.", cuda_status) }
  }
  pub fn set_from_host(&mut self, state: &[Complex32]) {
    let mut size = state.len();
    assert_eq!(size & (size - 1), 0, "State size is not a power of 2.");
    let qubits_number = {
      let mut counter: usize = 0;
      while size != 1 {
        size = size >> 1;
        counter += 1;
      }
      counter
    };
    assert_eq!(qubits_number, self.qubits_number, "Size of the given state does not match the size of the tensor.");
    let cuda_status = unsafe { set_from_host(self.state_ptr, state.as_ptr(), qubits_number) };
    if cuda_status != 0 { panic!("Cuda error with code {} occurred during copying from CPU to GPU.", cuda_status) }
  }
  pub fn get_conj_half(&self) -> Self {
    let mut conj_state_ptr = std::ptr::null_mut();
    let cuda_status = unsafe { get_state(&mut conj_state_ptr, self.qubits_number) };
    if cuda_status == 0
    {
      unsafe { conj_and_double(self.state_ptr, conj_state_ptr, self.qubits_number) };
      Self { state_ptr: conj_state_ptr, qubits_number: self.qubits_number }
    } else { panic!("Cuda error with code {} occurred during state allocation.", cuda_status)}
  }
  pub fn add(&mut self, other: Self) {
    assert_eq!(self.qubits_number, other.qubits_number, "Tensors have diferent sizes.");
    unsafe { add(other.state_ptr, self.state_ptr, self.qubits_number) };
  }
  pub fn get_cpu_state_copy(&self) -> Vec<Complex32> {
    let size = 1 << self.qubits_number;
    let mut state_vec = Vec::with_capacity(size);
    let copy_status = unsafe {
      state_vec.set_len(size);
      copy_to_host(self.state_ptr, state_vec.as_mut_ptr(), self.qubits_number)
    };
    assert_eq!(copy_status, 0, "Cuda error with code {} occurred during memcopy from the GPU to the CPU.", copy_status);
    state_vec
  }
  pub fn apply_q1_gate(&mut self, gate: &[Complex32], pos: usize) {
    assert_eq!(gate.len(), 4, "Incorrect len of the gate's buffer.");
    assert!(pos < self.qubits_number, "pos is out of the bound.");
    let cuda_status = unsafe { q1gate(self.state_ptr, gate.as_ptr(), pos, self.qubits_number) };
    assert_eq!(cuda_status, 0, "Cuda error with code {} occurred during a q1 gate application.", cuda_status);
  }
  pub fn apply_q1_gate_tr(&mut self, gate: &[Complex32], pos: usize) {
    let mut tr_gate = gate.to_owned();
    tr_gate.swap(1, 2);
    self.apply_q1_gate(&tr_gate[..], pos);
  }
  pub fn apply_q1_gate_conj_tr(&mut self, gate: &[Complex32], pos: usize) {
    let mut conj_tr_gate: Vec<Complex32> = gate.into_iter().map(|x| { x.conj() }).collect();
    conj_tr_gate.swap(1, 2);
    self.apply_q1_gate(&conj_tr_gate[..], pos);
  }
  pub fn apply_q2_gate(&mut self, gate: &[Complex32], pos2: usize, pos1: usize) {
    assert_eq!(gate.len(), 16, "Incorrect len of the gate's buffer.");
    assert!(pos1 != pos2, "pos1 and pos2 must be different.");
    assert!(pos1 < self.qubits_number, "pos1 is out of the bound.");
    assert!(pos2 < self.qubits_number, "pos2 is out of the bound.");
    let cuda_status = unsafe { q2gate(self.state_ptr, gate.as_ptr(), pos2, pos1, self.qubits_number) };
    assert_eq!(cuda_status, 0, "Cuda error with code {} occurred during a q1 gate application.", cuda_status);
  }
  pub fn apply_q2_gate_tr(&mut self, gate: &[Complex32], pos2: usize, pos1: usize) {
    let mut tr_gate = gate.to_owned();
    tr_gate.swap(1, 4); tr_gate.swap(2, 8); tr_gate.swap(6, 9);
    tr_gate.swap(3, 12); tr_gate.swap(7, 13); tr_gate.swap(11, 14);
    self.apply_q2_gate(&tr_gate[..], pos2, pos1);
  }
  pub fn apply_q2_gate_conj_tr(&mut self, gate: &[Complex32], pos2: usize, pos1: usize) {
    let mut conj_tr_gate: Vec<Complex32> = gate.into_iter().map(|x| { x.conj() }).collect();
    conj_tr_gate.swap(1, 4); conj_tr_gate.swap(2, 8); conj_tr_gate.swap(6, 9);
    conj_tr_gate.swap(3, 12); conj_tr_gate.swap(7, 13); conj_tr_gate.swap(11, 14);
    self.apply_q2_gate(&conj_tr_gate[..], pos2, pos1);
  }
  pub fn get_q1_density(&self, pos: usize) -> [Complex32; 4] {
    let mut density: [Complex32; 4] = Default::default();
    let cuda_status = unsafe { get_q1density(self.state_ptr, density.as_mut_ptr(), pos, self.qubits_number) };
    assert_eq!(cuda_status, 0, "Cuda error with code {} occurred during a q1 density matrix computation.", cuda_status);
    density
  }
  pub fn get_q2_density(&self, pos2: usize, pos1: usize) -> [Complex32; 16] {
    let mut density: [Complex32; 16] = Default::default();
    let cuda_status = unsafe { get_q2density(self.state_ptr, density.as_mut_ptr(), pos2, pos1, self.qubits_number) };
    assert_eq!(cuda_status, 0, "Cuda error with code {} occurred during a q1 density matrix computation.", cuda_status);
    density
  }
}

pub fn get_q1_grad(
  fwd: &QuantizedTensor,
  bwd: &QuantizedTensor,
  pos: usize,
) -> [Complex32; 4]
{
  assert_eq!(fwd.qubits_number, bwd.qubits_number, "fwd and bwd have different lengths.");
  assert!(pos < fwd.qubits_number, "pos out of range.");
  let mut grad: [Complex32; 4] = Default::default();
  unsafe { q1grad(fwd.state_ptr, bwd.state_ptr, grad.as_mut_ptr(), pos, bwd.qubits_number) };
  grad
}

pub fn get_q2_grad(
  fwd: &QuantizedTensor,
  bwd: &QuantizedTensor,
  pos2: usize,
  pos1: usize,
) -> [Complex32; 16]
{
  assert_eq!(fwd.qubits_number, bwd.qubits_number, "fwd and bwd have different lengths.");
  assert!(pos1 != pos2, "pos1 and pos2 must be different.");
  assert!(pos1 < fwd.qubits_number, "pos1 out of range.");
  assert!(pos2 < fwd.qubits_number, "pos2 out of range.");
  let mut grad: [Complex32; 16] = Default::default();
  unsafe { q2grad(fwd.state_ptr, bwd.state_ptr, grad.as_mut_ptr(), pos2, pos1, bwd.qubits_number) };
  grad
}

impl Drop for QuantizedTensor {
  fn drop(&mut self) {
      let cuda_status = unsafe { drop_state(self.state_ptr) };
      assert_eq!(cuda_status, 0, "Cuda error with code {} occurred during deallocation of a GPU memory.", cuda_status);
  }
}

impl Clone for QuantizedTensor {
  fn clone(&self) -> Self {
    let mut copy_state_ptr = std::ptr::null_mut();
    let cuda_status = unsafe { get_state(&mut copy_state_ptr, self.qubits_number) };
    if cuda_status == 0
    {
      unsafe { copy(self.state_ptr, copy_state_ptr, self.qubits_number) };
      Self { state_ptr: copy_state_ptr, qubits_number: self.qubits_number }
    } else { panic!("Cuda error with code {} occurred during state allocation.", cuda_status)}
  }
}

unsafe impl Send for QuantizedTensor {}

///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
  use super::*;
  use std::iter::zip;
  use super::super::common_gates::*;
  use super::super::test_utils::*;
  use ndarray::ArrayView1;
  use ndarray_rand::rand::random;
  use ndarray_einsum_beta::einsum;

  fn get_random_q1nonunitary() -> [Complex32; 4] {
    [
      Complex32::new(random(), random()), Complex32::new(random(), random()),
      Complex32::new(random(), random()), Complex32::new(random(), random()),
    ]
  }

  fn get_random_q2nonunitary() -> [Complex32; 16] {
    [
      Complex32::new(random(), random()), Complex32::new(random(), random()), Complex32::new(random(), random()), Complex32::new(random(), random()),
      Complex32::new(random(), random()), Complex32::new(random(), random()), Complex32::new(random(), random()), Complex32::new(random(), random()),
      Complex32::new(random(), random()), Complex32::new(random(), random()), Complex32::new(random(), random()), Complex32::new(random(), random()),
      Complex32::new(random(), random()), Complex32::new(random(), random()), Complex32::new(random(), random()), Complex32::new(random(), random()),
    ]
  }

  fn get_random_state_unnormalized(qubits_number: usize) -> Vec<Complex32> {
    (0..(1 << qubits_number))
      .map(|_| { Complex32::new(random(), random()) })
      .collect()
  }

  fn apply_q1_gate(state: &[Complex32], gate: &[Complex32], pos: usize) -> Vec<Complex32> {
    let mut size = state.len();
    assert_eq!(size & (size - 1), 0, "State size is not a power of 2.");
    let qubits_number = {
      let mut counter: usize = 0;
      while size != 1 {
        size = size >> 1;
        counter += 1;
      }
      counter
    };
    let state_view = ArrayView1::from(state);
    let state_view = state_view.into_shape((1 << (qubits_number - pos - 1), 2, 1 << pos)).unwrap();
    let gate_view = ArrayView1::from(gate);
    let gate_view = gate_view.into_shape((2, 2)).unwrap();
    einsum("iqk,jq->ijk", &[&state_view, &gate_view]).unwrap().iter().map(|x| { *x }).collect()
  }

  fn apply_q2_gate(state: &[Complex32], gate: &[Complex32], pos2: usize, pos1: usize) -> Vec<Complex32> {
    let mut size = state.len();
    assert_eq!(size & (size - 1), 0, "State size is not a power of 2.");
    let qubits_number = {
      let mut counter: usize = 0;
      while size != 1 {
        size = size >> 1;
        counter += 1;
      }
      counter
    };
    let state_view = ArrayView1::from(state);
    let (max_pos, min_pos) = if pos2 > pos1 { (pos2, pos1) } else { (pos1, pos2) };
    let state_view = state_view.into_shape((1 << (qubits_number - max_pos - 1), 2, 1 << (max_pos - min_pos - 1), 2, 1 << min_pos)).unwrap();
    let gate_view = ArrayView1::from(gate);
    let gate_view = gate_view.into_shape((2, 2, 2, 2)).unwrap();
    if pos2 > pos1 {
      einsum("iqkpm,jlqp->ijklm", &[&state_view, &gate_view]).unwrap().iter().map(|x| { *x }).collect()
    } else {
      einsum("iqkpm,ljpq->ijklm", &[&state_view, &gate_view]).unwrap().iter().map(|x| { *x }).collect()
    }
  }

  fn get_q1_density(state: &[Complex32], pos: usize) -> Vec<Complex32> {
    let mut size = state.len();
    assert_eq!(size & (size - 1), 0, "State size is not a power of 2.");
    let qubits_number = {
      let mut counter: usize = 0;
      while size != 1 {
        size = size >> 1;
        counter += 1;
      }
      counter
    };
    let state_view = ArrayView1::from(state);
    let state_view = state_view.into_shape((1 << (qubits_number - pos - 1), 2, 1 << pos)).unwrap();
    let mut state_conj = state_view.to_owned();
    state_conj.iter_mut().for_each(|x| { *x = x.conj(); });
    let state_conj_view = state_conj.view();
    einsum("iqj,ipj->qp", &[&state_view, &state_conj_view]).unwrap().iter().map(|x| { *x }).collect()
  }

  fn get_q2_density(state: &[Complex32], pos2: usize, pos1: usize) -> Vec<Complex32> {
    let mut size = state.len();
    assert_eq!(size & (size - 1), 0, "State size is not a power of 2.");
    let qubits_number = {
      let mut counter: usize = 0;
      while size != 1 {
        size = size >> 1;
        counter += 1;
      }
      counter
    };
    let state_view = ArrayView1::from(state);
    let (max_pos, min_pos) = if pos2 > pos1 { (pos2, pos1) } else { (pos1, pos2) };
    let state_view = state_view.into_shape((1 << (qubits_number - max_pos - 1), 2, 1 << (max_pos - min_pos - 1), 2, 1 << min_pos)).unwrap();
    let mut state_conj = state_view.to_owned();
    state_conj.iter_mut().for_each(|x| { *x = x.conj(); });
    let state_conj_view = state_conj.view();
    if pos2 > pos1 {
      einsum("iqkpm,irksm->qprs", &[&state_view, &state_conj_view]).unwrap().iter().map(|x| { *x }).collect()
    } else {
      einsum("iqkpm,irksm->pqsr", &[&state_view, &state_conj_view]).unwrap().iter().map(|x| { *x }).collect()
    }
  }

  fn get_q1_grad_test(fwd: &[Complex32], bwd: &[Complex32], pos: usize) -> Vec<Complex32> {
    assert_eq!(fwd.len(), bwd.len(), "Lengths of bwd and fwd are different.");
    let mut size = fwd.len();
    assert_eq!(size & (size - 1), 0, "State size is not a power of 2.");
    let qubits_number = {
      let mut counter: usize = 0;
      while size != 1 {
        size = size >> 1;
        counter += 1;
      }
      counter
    };
    let fwd_view = ArrayView1::from(fwd);
    let fwd_view = fwd_view.into_shape((1 << (qubits_number - pos - 1), 2, 1 << pos)).unwrap();
    let bwd_view = ArrayView1::from(bwd);
    let bwd_view = bwd_view.into_shape((1 << (qubits_number - pos - 1), 2, 1 << pos)).unwrap();
    einsum("iqj,ipj->qp", &[&bwd_view, &fwd_view]).unwrap().iter().map(|x| { *x }).collect()
  }

  fn get_q2_grad_test(fwd: &[Complex32], bwd: &[Complex32], pos2: usize, pos1: usize) -> Vec<Complex32> {
    assert_eq!(fwd.len(), bwd.len(), "Lengths of bwd and fwd are different.");
    let mut size = fwd.len();
    assert_eq!(size & (size - 1), 0, "State size is not a power of 2.");
    let qubits_number = {
      let mut counter: usize = 0;
      while size != 1 {
        size = size >> 1;
        counter += 1;
      }
      counter
    };
    let (max_pos, min_pos) = if pos2 > pos1 { (pos2, pos1) } else { (pos1, pos2) };
    let fwd_view = ArrayView1::from(fwd);
    let fwd_view = fwd_view.into_shape((1 << (qubits_number - max_pos - 1), 2, 1 << (max_pos - min_pos - 1), 2, 1 << min_pos)).unwrap();
    let bwd_view = ArrayView1::from(bwd);
    let bwd_view = bwd_view.into_shape((1 << (qubits_number - max_pos - 1), 2, 1 << (max_pos - min_pos - 1), 2, 1 << min_pos)).unwrap();
    if pos2 > pos1 {
      einsum("iqkpm,irksm->qprs", &[&bwd_view, &fwd_view]).unwrap().iter().map(|x| { *x }).collect()
    } else {
      einsum("iqkpm,irksm->pqsr", &[&bwd_view, &fwd_view]).unwrap().iter().map(|x| { *x }).collect()
    }
  }

  fn conj_half(state: &[Complex32]) -> Vec<Complex32> {
    state.into_iter().map(|x| { x.conj() / 2. }).collect()
  }

  fn add(src: &[Complex32], dst: &mut [Complex32]) {
    assert_eq!(src.len(), dst.len(), "src and dst lengths must be equal.");
    zip(src.into_iter(), dst.into_iter()).for_each(|(lhs, rhs)| {
      *rhs = *rhs + *lhs;
    })
  }

  #[test]
  fn test_q1gate() {
    let qubits_number = 17;
    let mut state = get_random_state_unnormalized(qubits_number);
    let mut vm = QuantizedTensor::new_from_host(&state[..]);
    for pos in 0..qubits_number {
      let q1random = get_random_q1nonunitary();
      state = apply_q1_gate(&state[..], &q1random[..], pos);
      vm.apply_q1_gate(&q1random[..], pos);
      let vm_state = vm.get_cpu_state_copy();
      cmp_complex_slices(&state[..], &vm_state[..], 1e-5);
    }
  }

  #[test]
  fn test_q2gate() {
    let qubits_number = 17;
    let iters_num = 20;
    let mut state = get_random_state_unnormalized(qubits_number);
    let mut vm = QuantizedTensor::new_from_host(&state[..]);
    for _ in 0..iters_num {
      let pos1: usize = random::<i32>() as usize % qubits_number;
      let pos2: usize = random::<i32>() as usize % qubits_number;
      if pos1 != pos2 {
        let q2random = get_random_q2nonunitary();
        state = apply_q2_gate(&state[..], &q2random[..], pos2, pos1);
        vm.apply_q2_gate(&q2random[..], pos2, pos1);
        let vm_state = vm.get_cpu_state_copy();
        cmp_complex_slices(&state[..], &vm_state[..], 1e-5);
      }
    }
  }

  #[test]
  fn test_ghz() {
    let qubits_number = 21;
    let mut vm = QuantizedTensor::new_standard(qubits_number);
    let hadamard = get_hadamard();
    let cnot = get_cnot();
    vm.apply_q1_gate(&hadamard[..], 0);
    for i in 0..(qubits_number - 1) {
      vm.apply_q2_gate(&cnot[..], i, i+1);
    }
    let state = vm.get_cpu_state_copy();
    let bell_ampl = Complex32::new(1. / f32::sqrt(2.), 0.);
    cmp_complex_slices(&state[..1], &[bell_ampl], 1e-5);
    cmp_complex_slices(&state[((1 << qubits_number) - 1)..], &[bell_ampl], 1e-5);
    cmp_complex_slices(
      &state[1..((1 << qubits_number) - 1)],
      &vec![Complex32::new(0., 0.); (1 << qubits_number) - 2][..],
      1e-5,
    );
  }

  #[test]
  fn test_q1density() {
    let qubits_number = 17;
    let state = get_random_state_unnormalized(qubits_number);
    let vm = QuantizedTensor::new_from_host(&state[..]);
    for i in 0..qubits_number {
      let dens = get_q1_density(&state[..], i);
      let vm_dens = vm.get_q1_density(i);
      cmp_complex_slices(&dens, &vm_dens, 1e-5);
    }
  }

  #[test]
  fn test_q2density() {
    let qubits_number = 17;
    let iters_num = 20;
    let state = get_random_state_unnormalized(qubits_number);
    let vm = QuantizedTensor::new_from_host(&state[..]);
    for _ in 0..iters_num {
      let pos1: usize = random::<i32>() as usize % qubits_number;
      let pos2: usize = random::<i32>() as usize % qubits_number;
      if pos1 != pos2 {
        let dens = get_q2_density(&state[..], pos2, pos1);
        let vm_dens = vm.get_q2_density(pos2, pos1);
        cmp_complex_slices(&dens, &vm_dens, 1e-5);
      }
    }
  }

  #[test]
  fn test_q1grad() {
    let qubits_number = 17;
    let fwd = get_random_state_unnormalized(qubits_number);
    let bwd = get_random_state_unnormalized(qubits_number);
    let fwd_vm = QuantizedTensor::new_from_host(&fwd[..]);
    let bwd_vm = QuantizedTensor::new_from_host(&bwd[..]);
    for pos in 0..qubits_number {
      let grad = get_q1_grad_test(&fwd[..], &bwd[..], pos);
      let vm_grad = get_q1_grad(&fwd_vm, &bwd_vm, pos);
      cmp_complex_slices(&grad, &vm_grad, 1e-5);
    }
  }

  #[test]
  fn test_q2grad() {
    let qubits_number = 17;
    let iters_num = 20;
    let fwd = get_random_state_unnormalized(qubits_number);
    let bwd = get_random_state_unnormalized(qubits_number);
    let fwd_vm = QuantizedTensor::new_from_host(&fwd[..]);
    let bwd_vm = QuantizedTensor::new_from_host(&bwd[..]);
    for _ in 0..iters_num {
      let pos1: usize = random::<i32>() as usize % qubits_number;
      let pos2: usize = random::<i32>() as usize % qubits_number;
      if pos1 != pos2 {
        let grad = get_q2_grad_test(&fwd[..], &bwd[..], pos2, pos1);
        let vm_grad = get_q2_grad(&fwd_vm, &bwd_vm, pos2, pos1);
        cmp_complex_slices(&grad, &vm_grad, 1e-5);
      }
    }
  }

  #[test]
  fn test_conj_half() {
    let qubits_number = 17;
    let state = get_random_state_unnormalized(qubits_number);
    let vm = QuantizedTensor::new_from_host(&state[..]);
    let conj1 = conj_half(&state);
    let conj2 = vm.get_conj_half().get_cpu_state_copy();
    cmp_complex_slices(&conj1[..], &conj2[..], 1e-5);
  }

  #[test]
  fn test_add() {
    let qubits_number = 17;
    let src = get_random_state_unnormalized(qubits_number);
    let mut dst = get_random_state_unnormalized(qubits_number);
    let vm_src = QuantizedTensor::new_from_host(&src[..]);
    let mut vm_dst = QuantizedTensor::new_from_host(&dst[..]);
    add(&src[..], &mut dst[..]);
    vm_dst.add(vm_src);
    cmp_complex_slices(&dst[..], &vm_dst.get_cpu_state_copy()[..], 1e-5);
  }
}