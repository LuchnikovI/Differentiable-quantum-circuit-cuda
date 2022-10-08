from jax.config import config
from jax.test_util import check_grads
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from circuit import AutoGradCircuit
from typing import List
from scipy.optimize import minimize
import numpy as np
from jax import jit, value_and_grad, random
from functools import partial
import time

# interaction gate from parameter
def zz(gamma: jnp.ndarray):
  return jnp.diag(jnp.array([
    jnp.exp(-1j * gamma),
    jnp.exp(1j * gamma),
    jnp.exp(1j * gamma),
    jnp.exp(-1j * gamma)
  ]))

# local gate from parameter
def x(beta: jnp.ndarray):
  return jnp.array([
    [jnp.cos(beta), -1j * jnp.sin(beta)],
    [-1j * jnp.sin(beta), jnp.cos(beta)]
  ])

# energy value
@partial(jit, static_argnums=0)
def energy(
  qubits_number: int,
  density_matrices: List[jnp.ndarray],
  h: jnp.ndarray,
) -> jnp.ndarray:
  e = 0.
  for dens in density_matrices:
    e += jnp.einsum("ij,ji", dens, h)
  return e.real

# gates from parameters
@partial(jit, static_argnums=1)
def params2gates(params: jnp.ndarray, qubits_number: int) -> List[jnp.ndarray]:
  layers_number = int(params.shape[0] / 2)
  gates = []
  for i in range(0, 2 * layers_number, 2):
    gates += qubits_number * [zz(params[i])]
    gates += qubits_number * [x(params[i + 1])]
  return gates

def run_vqcs_ising(
  qubits_number: int,
  layers_number: int,
  max_iters_number: int,
  magnetic_field: float,
):
  # initial state
  state = jnp.ones(2 ** qubits_number, dtype=jnp.complex64)
  state /= jnp.linalg.norm(state)

  # circuit
  c = AutoGradCircuit(qubits_number)
  c.set_state_from_vector(state)
  for _ in range(layers_number):
    for i in range(qubits_number-1):
      c.add_q2_var_gate(i, i+1)
    c.add_q2_var_gate(0, qubits_number-1)
    for i in range(qubits_number):
      c.add_q1_var_gate(i)
  for i in range(qubits_number-1):
    c.get_q2_dens_op_with_grad(i, i+1)
  c.get_q2_dens_op_with_grad(0, qubits_number-1)
  _, fwd_circ = c.build()

  # hamiltonian
  sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
  sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
  eye = jnp.eye(2, dtype=jnp.complex64)
  h = -jnp.tensordot(sz, sz, axes=0).transpose((0, 2, 1, 3)).reshape((4, 4)) -\
  0.5 * magnetic_field *\
  (jnp.tensordot(sx, eye, axes=0).transpose((0, 2, 1, 3)).reshape((4, 4)) +\
    jnp.tensordot(eye, sx, axes=0).transpose((0, 2, 1, 3)).reshape((4, 4)))

  # initial set of parameters
  key = random.PRNGKey(43)
  key, subkey = random.split(key)
  params = 0.1 * random.normal(subkey, (2 * layers_number,))

  # loss function
  def loss(params: jnp.ndarray) -> jnp.ndarray:
    gates = params2gates(params, qubits_number)
    dens_matrices = fwd_circ(gates, [])
    return energy(qubits_number, dens_matrices, h)

  # value and gradient
  _loss_grad = value_and_grad(loss)
  
  def loss_grad(params):
    v, g = _loss_grad(params)
    #print("Current energy: {}".format(v))
    return np.asarray(v, dtype=np.float64), np.asarray(g, dtype=np.float64)

  start = time.time()
  result = minimize(
    loss_grad,
    params,
    method = 'L-BFGS-B',
    jac = True,
    options={'maxiter': max_iters_number, 'disp': True},
  )
  end = time.time()
  e = result.fun
  exact_e = -2 * (1 / np.sin(np.pi / (2 * qubits_number)))  # for magnetic field = 1
  evals_num = result.nfev
  print('Exact energy: {}'.format(exact_e))
  print('Found energy: {}'.format(e))
  print('Relative error: {}'.format(np.abs(e - exact_e) / np.abs(exact_e)))
  print('Number of loss value and gradient calls: {}'.format(evals_num))
  print('Time per loss value and gradient call: {}'.format((end - start) / evals_num))

qubits_number = 22
layers_number = 22
max_iters_number = 300
magnetic_field = 1.

run_vqcs_ising(
  qubits_number,
  layers_number,
  max_iters_number,
  magnetic_field,
)