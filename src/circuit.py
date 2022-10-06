import numpy as np
from jax import custom_vjp
import jax.numpy as jnp
from typing import Callable, List, Tuple

from differentiable_circuit import Circuit

class AutoGradCircuit:

  def __init__(self, qubits_number: int):
    """Quantum circuit that supports automatic differentiation."""
    self.circuit = Circuit(qubits_number)
  
  def set_state_from_vector(
    self,
    vec: jnp.ndarray,
  ):
    """Set initial state from an array.
    Args:
      vec: jnp.ndarray, a state.
    """
    self.circuit.set_state_from_vector(np.asarray(vec))

  def add_q2_const_gate(self, pos2: int, pos1: int):
    """Adds a constant two-qubit gate to a circuit.
    Args:
      pos2: int, a position of a qubit that is considered as 'control' qubit,
      pos1: int, a position of a qubit that is conisdered as 'target' qubit.
    """
    self.circuit.add_q2_const_gate(pos2, pos1)

  def add_q2_var_gate(self, pos2: int, pos1: int):
    """Adds a variable two-qubit gate to a circuit.
    Args:
      pos2: int, a position of a qubit that is considered as 'control' qubit,
      pos1: int, a position of a qubit that is conisdered as 'target' qubit.
    """
    self.circuit.add_q2_var_gate(pos2, pos1)

  def add_q1_const_gate(self, pos: int):
    """Adds a constant one-qubit gate to a circuit.
    Args:
      pos: int, a position of a qubit that the gate is being applied to.
    """
    self.circuit.add_q1_const_gate(pos)

  def add_q1_var_gate(self, pos: int):
    """Adds a variable one-qubit gate to a circuit.
    Args:
      pos: int, a position of a qubit that the gate is being applied to.
    """
    self.circuit.add_q1_var_gate(pos)

  def get_q2_dens_op(self, pos2: int, pos1: int):
    """Adds an operation that evaluates a two-qubit density matrix
    when the circuit is being run.
    Args:
      pos2 and pos1: positions of qubits whose density matrix is being evaluated."""
    self.circuit.get_q2_dens_op(pos2, pos1)

  def get_q1_dens_op(self, pos: int):
    """Adds an operation that evaluates a one-qubit density matrix
    when the circuit is being run.
    Args:
      pos: position of a qubit whose density matrix is being evaluated."""
    self.circuit.get_q1_dens_op(pos)

  def get_q2_dens_op_with_grad(self, pos2: int, pos1: int):
    """Adds an operation that evaluates a two-qubit density matrix
    when the circuit is being run. At the backward pass it also
    takes gradient wrt the density matrix into account.
    Args:
      pos2 and pos1: positions of qubits whose density matrix is being evaluated."""
    self.circuit.get_q2_dens_op_with_grad(pos2, pos1)

  def get_q1_dens_op_with_grad(self, pos: int):
    """Adds an operation that evaluates a one-qubit density matrix
    when the circuit is being run. At the backward pass it also
    takes gradient wrt the density matrix into account.
    Args:
      pos: position of a qubit whose density matrix is being evaluated."""
    self.circuit.get_q1_dens_op_with_grad(pos)

  def build(
    self
  ) -> Tuple[
      Callable[[List[jnp.ndarray], List[jnp.ndarray]], List[np.ndarray]],
      Callable[[List[jnp.ndarray], List[jnp.ndarray]], List[np.ndarray]]
    ]:
    """Returns two functions. The first function runs the circuit given a
    list of constant and variable gates and evaluates all required density matrices.
    The second function evaluates only the density matrices whose gradients are
    necessary to take into account but it supports backpropagation for
    optimization purposes."""
    def simple_run(var_gates, const_gates):
      density_matrices = self.circuit.run(
        list(map(lambda x: np.asarray(x), const_gates)), 
        list(map(lambda x: np.asarray(x), var_gates)),
      )
      return density_matrices
    @custom_vjp
    def autodiff_run(var_gates, const_gates):
      density_matrices, _ = self.circuit.forward(
        list(map(lambda x: np.asarray(x), const_gates)), 
        list(map(lambda x: np.asarray(x), var_gates)),
      )
      return density_matrices
    def fwd_run(var_gates, const_gates):
      density_matrices, state = self.circuit.forward(
        list(map(lambda x: np.asarray(x), const_gates)), 
        list(map(lambda x: np.asarray(x), var_gates)),
      )
      return density_matrices, (const_gates, var_gates)
    def bwd_run(res, density_grads):
      const_gates, var_gates = res
      _, state = self.circuit.forward(
        list(map(lambda x: np.asarray(x), const_gates)), 
        list(map(lambda x: np.asarray(x), var_gates)),
      ) # real shit
      gates_grads = self.circuit.backward(
        state,
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