from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)
from jax.test_util import check_grads
import jax.numpy as jnp
from jax import random
from jax import grad
from qdc import AutoGradCircuit

def test_autodiff():
  layers = 10  # number of layers in a circuit
  qubits_number = 15  # number of qubits in a circuit
  eta = 1e-6  # perturbation step

  key = random.PRNGKey(42)

  # initial state
  initial_state = jnp.zeros(2 ** qubits_number, dtype=jnp.complex128)
  initial_state = initial_state.at[0].set(1.)

  # cnot gate
  cnot = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=jnp.complex128)

  # random unitary matrix generator
  def random_unitary(key: random.PRNGKey, size: int):
    u = random.normal(key, shape = (size, size, 2))
    u = u[..., 0] + 1j * u[..., 1]
    q, _ = jnp.linalg.qr(u)
    return q


  # random complex matrix generator
  def random_complex(key: random.PRNGKey, size: int):
    a = random.normal(key, shape = (size, size, 2))
    return a[..., 0] + 1j * a[..., 1]

  # here we define a circuit structure
  c = AutoGradCircuit(qubits_number)
  c.set_state_from_vector(initial_state)
  for _ in range(layers):
    for i in range(qubits_number):
      c.get_q1_dens_op_with_grad(i)
    for i in range(0, qubits_number-1, 2):
      c.get_q2_dens_op_with_grad(i+1, i)
    for i in range(qubits_number):
      c.add_q1_var_gate(i)
    for i in range(0, qubits_number-1, 2):
      c.add_q2_var_gate(i+1, i)
    for i in range(qubits_number):
      c.add_q1_const_gate(i)
    for i in range(1, qubits_number-1, 2):
      c.add_q2_const_gate(i+1, i)
    for i in range(i):
      c.get_q1_dens_op(i)
  for i in range(qubits_number):
      c.get_q1_dens_op(i)
  for i in range(0, qubits_number-1, 2):
      c.get_q2_dens_op(i+1, i)


  # this finction run a circuit and supports backprop.
  _, fwd_circ = c.build()

  def av_tsallis(var_gates, const_gates):
    density_matrices = fwd_circ(var_gates, const_gates)
    s = 0.
    for dens_matrix in density_matrices:
      s += (1 - jnp.einsum("ij,ji->", dens_matrix, dens_matrix)).real
    return s / len(density_matrices)


  # here we define circuit gates
  const_gates = []
  for _ in range(layers):
    key, subkey = random.split(key)
    const_gates += [random_unitary(k, 2) for k in random.split(subkey, qubits_number)]
    const_gates += int((qubits_number - 1) / 2) * [cnot]

  var_gates = []
  for _ in range(layers):
    key, subkey = random.split(key)
    var_gates += [random_unitary(k, 2) for k in random.split(subkey, qubits_number)]
    var_gates += int((qubits_number - 1) / 2) * [cnot]

  # here we define perturbated gates
  gates_perturbation = []
  for _ in range(layers):
    key, subkey = random.split(key)
    gates_perturbation += [random_complex(k, 2) for k in random.split(subkey, qubits_number)]
    key, subkey = random.split(key)
    gates_perturbation += [random_complex(k, 4) for k in random.split(subkey, int((qubits_number - 1) / 2))]
  var_gates_minus4eta = [lhs - 4 * eta * rhs for (lhs, rhs) in zip(var_gates, gates_perturbation)]
  var_gates_minus3eta = [lhs - 3 * eta * rhs for (lhs, rhs) in zip(var_gates, gates_perturbation)]
  var_gates_minus2eta = [lhs - 2 * eta * rhs for (lhs, rhs) in zip(var_gates, gates_perturbation)]
  var_gates_minus1eta = [lhs - 1 * eta * rhs for (lhs, rhs) in zip(var_gates, gates_perturbation)]
  var_gates_plus1eta = [lhs + 1 * eta * rhs for (lhs, rhs) in zip(var_gates, gates_perturbation)]
  var_gates_plus2eta = [lhs + 2 * eta * rhs for (lhs, rhs) in zip(var_gates, gates_perturbation)]
  var_gates_plus3eta = [lhs + 3 * eta * rhs for (lhs, rhs) in zip(var_gates, gates_perturbation)]
  var_gates_plus4eta = [lhs + 4 * eta * rhs for (lhs, rhs) in zip(var_gates, gates_perturbation)]

  # finite diff
  s_minus4eta = av_tsallis(var_gates_minus4eta, const_gates)
  s_minus3eta = av_tsallis(var_gates_minus3eta, const_gates)
  s_minus2eta = av_tsallis(var_gates_minus2eta, const_gates)
  s_minus1eta = av_tsallis(var_gates_minus1eta, const_gates)
  s_plus1eta = av_tsallis(var_gates_plus1eta, const_gates)
  s_plus2eta = av_tsallis(var_gates_plus2eta, const_gates)
  s_plus3eta = av_tsallis(var_gates_plus3eta, const_gates)
  s_plus4eta = av_tsallis(var_gates_plus4eta, const_gates)

  ds_finite = (1./280.) * s_minus4eta + (-1./280.) * s_plus4eta \
            + (-4./105.) * s_minus3eta + (4./105.) * s_plus3eta \
            + (1./5.) * s_minus2eta + (-1./5.) * s_plus2eta \
            + (-4./5.) * s_minus1eta + (4./5.) * s_plus1eta
  ds_finite /= eta

  # a gradient wrt gates
  gates_grad = grad(av_tsallis, argnums=0)(var_gates, const_gates)

  # here we calculate ds via gradients
  ds = 0.
  for (p, g) in zip(gates_perturbation, gates_grad):
    ds += jnp.tensordot(g, p, axes = [[0, 1], [0, 1]]).real

  assert(jnp.abs(ds - ds_finite) / min(jnp.abs(ds), jnp.abs(ds_finite)) < 1e-9)
