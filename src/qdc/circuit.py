import numpy as np
from jax import custom_vjp
import jax.numpy as jnp
from typing import Callable, List, Tuple

from quantum_differentiable_circuit import Circuit

class AutoGradCircuit:

  def __init__(self, qubits_number: int):
    """Quantum circuit with automatic differentiation."""
    self.circuit = Circuit(qubits_number)
  
  def set_state_from_vector(
    self,
    vec: jnp.ndarray,
  ):
    """Set initial state from a jax array.
    Args:
      vec: jnp.ndarray, an initial state.
    """
    self.circuit.set_state_from_vector(np.asarray(vec))

  def add_q2_const_gate(self, pos2: int, pos1: int):
    """Adds a constant two-qubit gate to the circuit.
    Args:
      pos2: int, a position of a qubit that is considered as 'control' qubit,
      pos1: int, a position of a qubit that is conisdered as 'target' qubit.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order).
    """
    self.circuit.add_q2_const_gate(pos2, pos1)
    
  def add_q2_const_gate_nonu(self, pos2: int, pos1: int):
    """Adds a constant non-unitary two-qubit gate to the circuit.
    Args:
      pos2: int, a position of a qubit that is considered as 'control' qubit,
      pos1: int, a position of a qubit that is conisdered as 'target' qubit.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order).
    """
    self.circuit.add_q2_const_gate_nonu(pos2, pos1)

  def add_q2_const_gate_diag(self, pos2: int, pos1: int):
    """Adds a diagonal constant two-qubit gate to the circuit.
    Args:
      pos2: int, a position of a qubit that is considered as 'control' qubit,
      pos1: int, a position of a qubit that is conisdered as 'target' qubit.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order).
    """
    self.circuit.add_q2_const_gate_diag(pos2, pos1)

  def add_q2_var_gate(self, pos2: int, pos1: int):
    """Adds a variable two-qubit gate to the circuit.
    Args:
      pos2: int, a position of a qubit that is considered as 'control' qubit,
      pos1: int, a position of a qubit that is conisdered as 'target' qubit.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order).
    """
    self.circuit.add_q2_var_gate(pos2, pos1)

  def add_q2_var_gate_nonu(self, pos2: int, pos1: int):
    """Adds a variable non-unitary two-qubit gate to the circuit.
    Args:
      pos2: int, a position of a qubit that is considered as 'control' qubit,
      pos1: int, a position of a qubit that is conisdered as 'target' qubit.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order).
    """
    self.circuit.add_q2_var_gate_nonu(pos2, pos1)

  def add_q2_var_gate_diag(self, pos2: int, pos1: int):
    """Adds a diagoneal variable two-qubit gate to the circuit.
    Args:
      pos2: int, a position of a qubit that is considered as 'control' qubit,
      pos1: int, a position of a qubit that is conisdered as 'target' qubit.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order).
    """
    self.circuit.add_q2_var_gate_diag(pos2, pos1)

  def add_q1_const_gate(self, pos: int):
    """Adds a constant one-qubit gate to the circuit.
    Args:
      pos: int, a position of a qubit that the gate is being applied to.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order).
    """
    self.circuit.add_q1_const_gate(pos)

  def add_q1_const_gate_nonu(self, pos: int):
    """Adds a constant non-unitary one-qubit gate to the circuit.
    Args:
      pos: int, a position of a qubit that the gate is being applied to.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order).
    """
    self.circuit.add_q1_const_gate_nonu(pos)

  def add_q1_var_gate(self, pos: int):
    """Adds a variable one-qubit gate to the circuit.
    Args:
      pos: int, a position of a qubit that the gate is being applied to.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order).
    """
    self.circuit.add_q1_var_gate(pos)

  def add_q1_var_gate_nonu(self, pos: int):
    """Adds a variable non-unitary one-qubit gate to the circuit.
    Args:
      pos: int, a position of a qubit that the gate is being applied to.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order).
    """
    self.circuit.add_q1_var_gate_nonu(pos)

  def get_q2_dens_op(self, pos2: int, pos1: int):
    """Adds an operation to the circuit that evaluates a two-qubit density matrix
    when the circuit is being run.
    Args:
      pos2 and pos1: positions of qubits whose density matrix is being evaluated.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order)."""
    self.circuit.get_q2_dens_op(pos2, pos1)

  def get_q1_dens_op(self, pos: int):
    """Adds an operation to the circuit that evaluates a one-qubit density matrix
    when the circuit is being run.
    Args:
      pos: position of a qubit whose density matrix is being evaluated.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order)."""
    self.circuit.get_q1_dens_op(pos)

  def get_q2_dens_op_with_grad(self, pos2: int, pos1: int):
    """Adds an operation to the circuit that evaluates a two-qubit density matrix
    when the circuit is being run. At the backward propagation stage
    the gradient wrt to the given density matrix is being propagated
    through the circuit.
    Args:
      pos2 and pos1: positions of qubits whose density matrix is being evaluated.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order)."""
    self.circuit.get_q2_dens_op_with_grad(pos2, pos1)

  def get_q1_dens_op_with_grad(self, pos: int):
    """Adds an operation that evaluates a one-qubit density matrix
    when the circuit is being run. At the backward propagation stage
    the gradient wrt to the given density matrix is being propagated
    through the circuit.
    Args:
      pos: position of a qubit whose density matrix is being evaluated.
    Qubits are enumerated starting from the innermost one (the inverse numpy/jax
    order)."""
    self.circuit.get_q1_dens_op_with_grad(pos)

  def build(
    self
  ) -> Tuple[
      Callable[[List[jnp.ndarray], List[jnp.ndarray]], List[np.ndarray]],
      Callable[[List[jnp.ndarray], List[jnp.ndarray]], List[np.ndarray]]
    ]:
    """Returns two functions. The first function runs the circuit given a
    list of constant and variable gates and evaluates all required density matrices. The
    first function does not support backward pass.
    The second function evaluates only those density matrices which take part in the gradients
    backward pass. The second function supports backard pass."""
    def simple_run(var_gates, const_gates):
      density_matrices = self.circuit.run(
        list(map(lambda x: np.asarray(x), const_gates)), 
        list(map(lambda x: np.asarray(x), var_gates)),
      )
      return density_matrices
    @custom_vjp
    def autodiff_run(var_gates, const_gates):
      density_matrices = self.circuit.forward(
        list(map(lambda x: np.asarray(x), const_gates)), 
        list(map(lambda x: np.asarray(x), var_gates)),
      )
      return density_matrices
    def fwd_run(var_gates, const_gates):
      density_matrices = self.circuit.forward(
        list(map(lambda x: np.asarray(x), const_gates)), 
        list(map(lambda x: np.asarray(x), var_gates)),
      )
      return density_matrices, (const_gates, var_gates)
    def bwd_run(res, density_grads):
      const_gates, var_gates = res
      gates_grads = self.circuit.backward(
        list(map(lambda x: np.asarray(x).conj(), density_grads)),
        list(map(lambda x: np.asarray(x), const_gates)), 
        list(map(lambda x: np.asarray(x), var_gates)),
      )
      return gates_grads, None
    autodiff_run.defvjp(
      fwd_run,
      bwd_run,
    )
    return simple_run, autodiff_run
