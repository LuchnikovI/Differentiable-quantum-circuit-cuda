from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from qdc import AutoGradCircuit

def test_ghz():
  qubits_number = 21  # number of qubits in a circuit

  # cnot gate
  cnot = jnp.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], dtype=jnp.complex128)
  hadamard = jnp.array([1, 1, 1, -1], dtype=jnp.complex128) * (1 / jnp.sqrt(2))

  c = AutoGradCircuit(qubits_number)

  c.add_q1_const_gate(0)
  for i in range(qubits_number-1):
    c.get_q2_dens_op_with_grad(i, i+1)
  for i in range(qubits_number):
    c.get_q1_dens_op_with_grad(i)
  for i in range(qubits_number-1):
    c.add_q2_const_gate(i, i+1)
  for i in range(qubits_number):
    c.get_q1_dens_op(i)
  for i in range(qubits_number-1):
    c.get_q2_dens_op(i, i+1)

  simple_run, autodiff_run = c.build()

  all_density_matrices = simple_run([], [hadamard] + (qubits_number - 1) * [cnot])
  autodiff_density_matrices = autodiff_run([], [hadamard] + (qubits_number - 1) * [cnot])
  assert(len(all_density_matrices) == 2 * qubits_number + 2 * (qubits_number - 1))
  assert(len(autodiff_density_matrices) == qubits_number + (qubits_number - 1))
  for lhs, rhs in zip(all_density_matrices[:qubits_number + (qubits_number - 1)], autodiff_density_matrices):
    assert(jnp.isclose(lhs, rhs).all())

  first_psi = jnp.tensordot(jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2)]), jnp.array([1., 0.]), axes=0).reshape((4,))
  first_dens = jnp.tensordot(first_psi, first_psi.conj(), axes=0).reshape((4, 4))
  assert(jnp.isclose(first_dens, all_density_matrices[0]).all())

  second_dens = jnp.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
  for dens in all_density_matrices[1:(qubits_number-1)]:
    assert(jnp.isclose(dens, second_dens).all())

  superposition_dens = jnp.array([[0.5, 0.5], [0.5, 0.5]])
  assert(jnp.isclose(superposition_dens, all_density_matrices[(qubits_number-1)]).all())

  up_spin_dens = jnp.array([[1, 0], [0, 0]])
  for dens in all_density_matrices[qubits_number:(2 * qubits_number - 1)]:
    assert(jnp.isclose(dens, up_spin_dens).all())
    
  one_qubit_mixed = jnp.array([[0.5, 0.], [0., 0.5]])
  for dens in all_density_matrices[(2 * qubits_number - 1):(3 * qubits_number - 1)]:
    assert(jnp.isclose(dens, one_qubit_mixed).all())

  two_qubit_partial_mixed = jnp.array([[0.5, 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., .0, 0., 0.5]])
  for dens in all_density_matrices[(3 * qubits_number - 1):]:
    assert(jnp.isclose(dens, two_qubit_partial_mixed).all())
