use std::{collections::VecDeque, ops::{DerefMut, Deref}};
use pyo3::prelude::{
  pyclass,
  pymethods,
  Python,
  pymodule,
  PyResult,
  PyModule,
};
use numpy::{
  PyReadonlyArray1,
  PyReadonlyArray2,
  PyArray2,
  PyArray1,
};
use num_complex;

#[cfg(not(feature = "f64"))]
use std::os::raw::c_float;

#[cfg(feature = "f64")]
use std::os::raw::c_double;

use std::iter::zip;

use super::{
  QuantizedTensor,
  get_q1_grad,
  get_q2_grad,
  get_q2_grad_diag,
  data_transfer,
};

#[cfg(not(feature = "f64"))]
type Complex = num_complex::Complex<c_float>;
#[cfg(feature = "f64")]
type Complex = num_complex::Complex<c_double>;

fn np_array_from_slice<'py>(slice: &[Complex], py: Python<'py>) -> &'py PyArray1<Complex> {
  let size = slice.len();
  let nparr = unsafe { PyArray1::new(py, size, false) };
  zip(unsafe { nparr.as_slice_mut().unwrap().into_iter() }, slice).for_each(|(dst, src)| {
    *dst = *src;
  });
  nparr
}

fn np_matrix_from_slice<'py>(slice: &[Complex], py: Python<'py>) -> &'py PyArray2<Complex> {
  let size = (slice.len() as f64).sqrt() as usize;
  assert_eq!(size * size, slice.len(), "Matrix is not square.");
  let nparr = unsafe { PyArray2::new(py, (size, size), false) };
  zip(unsafe { nparr.as_slice_mut().unwrap().into_iter() }, slice).for_each(|(dst, src)| {
    *dst = *src;
  });
  nparr
}

enum Instruction {
  ConstQ2Gate((usize, usize)),
  VarQ2Gate((usize, usize)),
  ConstQ2GateDiag((usize, usize)),
  VarQ2GateDiag((usize, usize)),
  ConstQ1Gate(usize),
  VarQ1Gate(usize),
  Q2Density((usize, usize)),
  Q1Density(usize),
  DiffQ2Density((usize, usize)),
  DiffQ1Density(usize),
}

#[pyclass]
pub struct TensorPyWrapper(QuantizedTensor);

impl Deref for TensorPyWrapper {
  type Target = QuantizedTensor;
  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl DerefMut for TensorPyWrapper {
  fn deref_mut(&mut self) -> &mut Self::Target {
      &mut self.0
  }
}

#[pyclass]
pub struct Circuit {
  instructions: Vec<Instruction>,
  initial_state: QuantizedTensor,
  state: QuantizedTensor,
}

#[pymethods]
impl Circuit {
  #[new]
  fn new<'py>(qubits_number: usize) -> Self {
    let state = QuantizedTensor::new_standard(qubits_number);
    Self {
      instructions: Vec::new(),
      initial_state: state.clone(),
      state: state,
    }
  }
  fn set_state_from_vector(&mut self, vector: PyReadonlyArray1<Complex>) {
    self.initial_state.set_from_host(vector.as_slice().unwrap());
  }

  fn add_q2_const_gate(&mut self, pos2: usize, pos1: usize) {
    self.instructions.push(Instruction::ConstQ2Gate((pos2, pos1)))
  }

  fn add_q2_const_gate_diag(&mut self, pos2: usize, pos1: usize) {
    self.instructions.push(Instruction::ConstQ2GateDiag((pos2, pos1)))
  }

  fn add_q2_var_gate(&mut self, pos2: usize, pos1: usize) {
    self.instructions.push(Instruction::VarQ2Gate((pos2, pos1)))
  }

  fn add_q2_var_gate_diag(&mut self, pos2: usize, pos1: usize) {
    self.instructions.push(Instruction::VarQ2GateDiag((pos2, pos1)))
  }

  fn add_q1_const_gate(&mut self, pos: usize) {
    self.instructions.push(Instruction::ConstQ1Gate(pos))
  }

  fn add_q1_var_gate(&mut self, pos: usize) {
    self.instructions.push(Instruction::VarQ1Gate(pos))
  }

  fn get_q2_dens_op(&mut self, pos2: usize, pos1: usize) {
    self.instructions.push(Instruction::Q2Density((pos2, pos1)))
  }

  fn get_q1_dens_op(&mut self, pos: usize) {
    self.instructions.push(Instruction::Q1Density(pos))
  }

  fn get_q2_dens_op_with_grad(&mut self, pos2: usize, pos1: usize) {
    self.instructions.push(Instruction::DiffQ2Density((pos2, pos1)))
  }

  fn get_q1_dens_op_with_grad(&mut self, pos: usize) {
    self.instructions.push(Instruction::DiffQ1Density(pos))
  }

  fn run<'py>(
    &mut self,
    const_gates: Vec<PyReadonlyArray1<Complex>>,
    var_gates: Vec<PyReadonlyArray1<Complex>>,
    py: Python<'py>,
  ) -> Vec<&'py PyArray2<Complex>> {
    assert!(!self.instructions.is_empty(), "The circuit is empty.");
    let mut density_matrices = Vec::new();
    let mut const_gates = VecDeque::from(const_gates);
    let mut var_gates = VecDeque::from(var_gates);
    data_transfer(&self.initial_state, &mut self.state);
    for inst in &self.instructions {
      match inst {
        Instruction::ConstQ1Gate(pos) => {
          let gate = const_gates.pop_front().expect("The number of constant gates is less than required.");
          self.state.apply_q1_gate(gate.as_slice().expect("Gate is not contiguous."), *pos);
        },
        Instruction::ConstQ2Gate((pos2, pos1)) => {
          let gate = const_gates.pop_front().expect("The number of constant gates is less than required.");
          self.state.apply_q2_gate(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
        },
        Instruction::ConstQ2GateDiag((pos2, pos1)) => {
          let gate = const_gates.pop_front().expect("The number of constant gates is less than required.");
          self.state.apply_q2_gate_diag(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
        },
        Instruction::VarQ1Gate(pos) => {
          let gate = var_gates.pop_front().expect("The number of variable gates is less than required.");
          self.state.apply_q1_gate(gate.as_slice().expect("Gate is not contiguous."), *pos);
        },
        Instruction::VarQ2Gate((pos2, pos1)) => {
          let gate = var_gates.pop_front().expect("The number of variable gates is less than required.");
          self.state.apply_q2_gate(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
        },
        Instruction::VarQ2GateDiag((pos2, pos1)) => {
          let gate = var_gates.pop_front().expect("The number of constant gates is less than required.");
          self.state.apply_q2_gate_diag(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
        },
        Instruction::Q1Density(pos) | Instruction::DiffQ1Density(pos)=> {
          density_matrices.push(np_matrix_from_slice(&self.state.get_q1_density(*pos)[..], py));
        },
        Instruction::Q2Density((pos2, pos1)) | Instruction::DiffQ2Density((pos2, pos1)) => {
          density_matrices.push(np_matrix_from_slice(&self.state.get_q2_density(*pos2, *pos1)[..], py));
        },
      }
    }
    if !const_gates.is_empty() { panic!("Number of constant gates is more than required.") };
    if !var_gates.is_empty() { panic!("Number of variable gates is more than required.") };
    density_matrices
  }

  fn forward<'py>(
    &mut self,
    const_gates: Vec<PyReadonlyArray1<Complex>>,
    var_gates: Vec<PyReadonlyArray1<Complex>>,
    py: Python<'py>,
  ) -> Vec<&'py PyArray2<Complex>>
  {
    assert!(!self.instructions.is_empty(), "The circuit is empty.");
    let mut density_matrices = Vec::new();
    let mut const_gates = VecDeque::from(const_gates);
    let mut var_gates = VecDeque::from(var_gates);
    data_transfer(&self.initial_state, &mut self.state);
    for inst in &self.instructions {
      match inst {
        Instruction::ConstQ1Gate(pos) => {
          let gate = const_gates.pop_front().expect("The number of constant gates is less than required.");
          self.state.apply_q1_gate(gate.as_slice().expect("Gate is not contiguous."), *pos);
        },
        Instruction::ConstQ2Gate((pos2, pos1)) => {
          let gate = const_gates.pop_front().expect("The number of constant gates is less than required.");
          self.state.apply_q2_gate(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
        },
        Instruction::ConstQ2GateDiag((pos2, pos1)) => {
          let gate = const_gates.pop_front().expect("The number of constant gates is less than required.");
          self.state.apply_q2_gate_diag(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
        },
        Instruction::VarQ1Gate(pos) => {
          let gate = var_gates.pop_front().expect("The number of variable gates is less than required.");
          self.state.apply_q1_gate(gate.as_slice().expect("Gate is not contiguous."), *pos);
        },
        Instruction::VarQ2Gate((pos2, pos1)) => {
          let gate = var_gates.pop_front().expect("The number of variable gates is less than required.");
          self.state.apply_q2_gate(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
        },
        Instruction::VarQ2GateDiag((pos2, pos1)) => {
          let gate = var_gates.pop_front().expect("The number of constant gates is less than required.");
          self.state.apply_q2_gate_diag(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
        },
        Instruction::DiffQ1Density(pos)=> {
          density_matrices.push(np_matrix_from_slice(&self.state.get_q1_density(*pos)[..], py));
        },
        Instruction::DiffQ2Density((pos2, pos1)) => {
          density_matrices.push(np_matrix_from_slice(&self.state.get_q2_density(*pos2, *pos1)[..], py));
        },
        _ => {},
      }
    }
    if !const_gates.is_empty() { panic!("Number of constant gates is more than required.") };
    if !var_gates.is_empty() { panic!("Number of variable gates is more than required.") };
    density_matrices
  }

  fn backward<'py>(
    &mut self,
    mut grads_wrt_density: Vec<PyReadonlyArray2<Complex>>,
    mut const_gates: Vec<PyReadonlyArray1<Complex>>,
    mut var_gates: Vec<PyReadonlyArray1<Complex>>,
    py: Python<'py>,
  ) -> Vec<&'py PyArray1<Complex>>
  {
    assert!(!self.instructions.is_empty(), "The circuit is empty.");
    let fwd = &mut self.state;
    let mut bwd_option: Option<QuantizedTensor> = None;
    let mut grads_wrt_gates = VecDeque::new();
    for inst in self.instructions.iter().rev() {
      match inst {
        Instruction::ConstQ1Gate(pos) => {
          let gate = const_gates.pop().expect("The number of gates is less than required.");
          fwd.apply_q1_gate_conj_tr(gate.as_slice().expect("Gate is not contiguous."), *pos);
          if let Some(mut bwd) = bwd_option.take() {
            bwd.apply_q1_gate_tr(gate.as_slice().expect("Gate is not contiguous."), *pos);
            bwd_option = Some(bwd);
          }
        },
        Instruction::ConstQ2Gate((pos2, pos1)) => {
          let gate = const_gates.pop().expect("The number of gates is less than required.");
          fwd.apply_q2_gate_conj_tr(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
          if let Some(mut bwd) = bwd_option.take() {
            bwd.apply_q2_gate_tr(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
            bwd_option = Some(bwd);
          }
        },
        Instruction::ConstQ2GateDiag((pos2, pos1)) => {
          let gate = const_gates.pop().expect("The number of gates is less than required.");
          fwd.apply_q2_gate_diag_conj(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
          if let Some(mut bwd) = bwd_option.take() {
            bwd.apply_q2_gate_diag(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
            bwd_option = Some(bwd);
          }
        },
        Instruction::VarQ1Gate(pos) => {
          let gate = var_gates.pop().expect("The number of gates is less than required.");
          fwd.apply_q1_gate_conj_tr(gate.as_slice().expect("Gate is not contiguous."), *pos);
          if let Some(mut bwd) = bwd_option.take() {
            grads_wrt_gates.push_front(np_array_from_slice(&get_q1_grad(&fwd, &bwd, *pos)[..], py));
            bwd.apply_q1_gate_tr(gate.as_slice().expect("Gate is not contiguous."), *pos);
            bwd_option = Some(bwd);
          } else {
            grads_wrt_gates.push_front(np_array_from_slice(&[
              Complex::new(0., 0.), Complex::new(0., 0.),
              Complex::new(0., 0.), Complex::new(0., 0.),
            ], py));
          }
        },
        Instruction::VarQ2Gate((pos2, pos1)) => {
          let gate = var_gates.pop().expect("The number of gates is less than required.");
          fwd.apply_q2_gate_conj_tr(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
          if let Some(mut bwd) = bwd_option.take() {
            grads_wrt_gates.push_front(np_array_from_slice(&get_q2_grad(&fwd, &bwd, *pos2, *pos1)[..], py));
            bwd.apply_q2_gate_tr(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
            bwd_option = Some(bwd);
          } else {
            grads_wrt_gates.push_front(np_array_from_slice(&[
              Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
              Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
              Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
              Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            ], py));
          }
        },
        Instruction::VarQ2GateDiag((pos2, pos1)) => {
          let gate = var_gates.pop().expect("The number of gates is less than required.");
          fwd.apply_q2_gate_diag_conj(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
          if let Some(mut bwd) = bwd_option.take() {
            grads_wrt_gates.push_front(np_array_from_slice(&get_q2_grad_diag(&fwd, &bwd, *pos2, *pos1)[..], py));
            bwd.apply_q2_gate_diag(gate.as_slice().expect("Gate is not contiguous."), *pos2, *pos1);
            bwd_option = Some(bwd);
          } else {
            grads_wrt_gates.push_front(np_array_from_slice(&[
              Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            ], py));
          }
        },
        Instruction::DiffQ1Density(pos) => {
          if let Some(mut bwd) = bwd_option.take() {
            let grad = grads_wrt_density.pop().expect("The number of gradients wrt density matrices is less than required.");
            let mut bwd_addition = fwd.conj_and_double();
            bwd_addition.apply_q1_gate_tr(grad.as_slice().expect("Gradient is not contiguous."), *pos);
            bwd.add(bwd_addition);
            bwd_option = Some(bwd);
          } else {
            let grad = grads_wrt_density.pop().expect("The number of gradients wrt density matrices is less than required.");
            let mut bwd_addition = fwd.conj_and_double();
            bwd_addition.apply_q1_gate_tr(grad.as_slice().expect("Gradient is not contiguous."), *pos);
            bwd_option = Some(bwd_addition);
          }
        },
        Instruction::DiffQ2Density((pos2, pos1)) => {
          if let Some(mut bwd) = bwd_option.take() {
            let grad = grads_wrt_density.pop().expect("The number of gradients wrt density matrices is less than required.");
            let mut bwd_addition = fwd.conj_and_double();
            bwd_addition.apply_q2_gate_tr(grad.as_slice().expect("Gradient is not contiguous."), *pos2, *pos1);
            bwd.add(bwd_addition);
            bwd_option = Some(bwd);
          } else {
            let grad = grads_wrt_density.pop().expect("The number of gradients wrt density matrices is less than required.");
            let mut bwd_addition = fwd.conj_and_double();
            bwd_addition.apply_q2_gate_tr(grad.as_slice().expect("Gradient is not contiguous."), *pos2, *pos1);
            bwd_option = Some(bwd_addition);
          }
        },
        Instruction::Q1Density(_) => {},
        Instruction::Q2Density((_, _)) => {},
      }
    }
    if !const_gates.is_empty() { panic!("Number of constant gates is more than required.") }
    if !var_gates.is_empty() { panic!("Number of constant gates is more than required.") }
    if !grads_wrt_density.is_empty() { panic!("Number of gradients wrt density matrices is more than required.") }
    Vec::from(grads_wrt_gates)
  }
}

#[pymodule]
fn quantum_differentiable_circuit(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Circuit>()?;
    Ok(())
}