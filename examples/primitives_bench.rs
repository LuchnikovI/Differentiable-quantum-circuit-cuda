/*use std::time::Instant;
use num_complex::Complex32;
use ndarray_rand::rand::random; 
use quantum_differentiable_emulator_cuda::{
  QuantizedTensor,
  get_q1_grad,
  get_q2_grad,
};

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

fn main() {
  let qubits_number = 28;
  let q1g = get_random_q1nonunitary();
  let q2g = get_random_q2nonunitary();
  {
    let state = QuantizedTensor::new_standard(qubits_number); // first cudaMalloc takes more time
  }
  let start = Instant::now();
  let mut state = QuantizedTensor::new_standard(qubits_number);
  println!("For {} qubits one has the following benchmarking results.
!!Note, that these benchmarks are not reliable and are used only for qualitative
performance evaluation!!\n", qubits_number
  );
  let duration = start.elapsed();
  println!("Standard state preparation takes: {:?}", (duration.as_nanos() as f64) / 1e9);
  let start = Instant::now();
  for pos in 0..qubits_number {
    state.apply_q1_gate(&q1g, pos);
  }
  let duration = start.elapsed();
  println!("A one-qubit gate execution takes: {:?} secs", (duration.as_nanos() as f64) / ((qubits_number as f64) * 1e9));
  let start = Instant::now();
  let mut counter = 0;
  for _ in 0..20 {
    let pos1: usize = random::<i32>() as usize % qubits_number;
    let pos2: usize = random::<i32>() as usize % qubits_number;
    if pos2 != pos1 {
      counter += 1;
      state.apply_q2_gate(&q2g, pos2, pos1);
    }
  }
  let duration = start.elapsed();
  println!("A two-qubit gate execution takes: {:?} secs", (duration.as_nanos() as f64) / ((counter as f64) * 1e9));
  let start = Instant::now();
  for pos in 0..qubits_number {
    let dens = state.get_q1_density(pos);
  }
  let duration = start.elapsed();
  println!("A one-qubit density matrix evaluation takes: {:?} secs", (duration.as_nanos() as f64) / ((qubits_number as f64) * 1e9));
  let start = Instant::now();
  let mut counter = 0;
  for _ in 0..20 {
    let pos1: usize = random::<i32>() as usize % qubits_number;
    let pos2: usize = random::<i32>() as usize % qubits_number;
    if pos2 != pos1 {
      counter += 1;
      state.get_q2_density(pos2, pos1);
    }
  }
  let duration = start.elapsed();
  println!("A two-qubit density matrix evaluation takes: {:?} secs", (duration.as_nanos() as f64) / ((counter as f64) * 1e9));
  let other_state = QuantizedTensor::new_standard(qubits_number);
  let start = Instant::now();
  for pos in 0..qubits_number {
    let grad = get_q1_grad(&state, &other_state, pos);
  }
  let duration = start.elapsed();
  println!("A one-qubit gate gradient evaluation takes: {:?} secs", (duration.as_nanos() as f64) / ((qubits_number as f64) * 1e9));
  let start = Instant::now();
  let mut counter = 0;
  for _ in 0..20 {
    let pos1: usize = random::<i32>() as usize % qubits_number;
    let pos2: usize = random::<i32>() as usize % qubits_number;
    if pos2 != pos1 {
      counter += 1;
      let grad = get_q2_grad(&state, &other_state, pos2, pos1);
    }
  }
  let duration = start.elapsed();
  println!("A two-qubit gate gradient evaluation takes: {:?} secs", (duration.as_nanos() as f64) / ((counter as f64) * 1e9));
  let start = Instant::now();
  state.add(other_state);
  let duration = start.elapsed();
  println!("Add function evaluation takes: {:?} secs", (duration.as_nanos() as f64) / 1e9);
  let start = Instant::now();
  let conj_half = state.get_conj_half();
  let duration = start.elapsed();
  println!("Get_conj_half function evaluation takes: {:?} secs", (duration.as_nanos() as f64) / 1e9);
  let start = Instant::now();
  {
    let state = state;
  }
  let duration = start.elapsed();
  println!("State drop takes: {:?} secs", (duration.as_nanos() as f64) / 1e9);
}*/
fn main() {} // this is to shut up the rust analyzer