import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin")
from jax.config import config
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from qdc import AutoGradCircuit
from typing import List
from scipy.optimize import minimize
import numpy as np
from jax import jit, value_and_grad, random
from functools import partial
import time

# zz interaction two-qubit gate from parameter
def zz(gamma: jnp.ndarray):
  return jnp.array([
    jnp.exp(-1j * gamma),
    jnp.exp(1j * gamma),
    jnp.exp(1j * gamma),
    jnp.exp(-1j * gamma)
  ])

# x one-qubit gate from parameter
def x(beta: jnp.ndarray):
  return jnp.array(
    [jnp.cos(beta), -1j * jnp.sin(beta),
    -1j * jnp.sin(beta), jnp.cos(beta)]
    )

# energy value from a two-qubit hamiltonian term and two-qubit density matrices
@jit
def energy(
  density_matrices: List[jnp.ndarray],
  h: jnp.ndarray,
) -> jnp.ndarray:
  e = 0.
  for dens in density_matrices:
    e += jnp.einsum("ij,ji", dens, h)
  return e.real

# gates from a vector of parameters
@partial(jit, static_argnums=1)
def params2gates(params: jnp.ndarray, qubits_number: int) -> List[jnp.ndarray]:
  layers_number = int(params.shape[0] / 2)
  gates = []
  for i in range(0, 2 * layers_number, 2):
    gates += qubits_number * [zz(params[i])]
    gates += qubits_number * [x(params[i + 1])]
  return gates

# parameters of the problem
qubits_number = 22
layers_number = 22  # number of alternatin layers (ZZ interaction, X magnetic field)
max_iters_number = 300
magnetic_field = 1.  # the phase transition point

# initial state (all Bloch vectors point along the X axis)
state = jnp.ones(2 ** qubits_number, dtype=jnp.complex64)
state /= jnp.linalg.norm(state)

# Here we define and build a circuit
c = AutoGradCircuit(qubits_number)

# set initial state
c.set_state_from_vector(state)

# add gates to the circuit in a loop
for _ in range(layers_number):
  # a layer of two-qubit interaction gates
  for i in range(qubits_number-1):
    c.add_q2_var_gate_diag(i, i+1)
  c.add_q2_var_gate_diag(0, qubits_number-1)
  # a layer of one-qubit gates
  for i in range(qubits_number):
    c.add_q1_var_gate(i)
# finally one adds requests to evaluate all two-qubit density matrices of neighboring qubits
for i in range(qubits_number-1):
  c.get_q2_dens_op_with_grad(i, i+1)
c.get_q2_dens_op_with_grad(0, qubits_number-1)

# fwd_circ is a function that takes a list of variable gates,
# a list of constant gates and returns all requested density matrices.
# It supports the reverse-mode automatic differentiation.
_, fwd_circ = c.build()

# two-qubit hamiltonian (Transverse-field Ising model)
sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
eye = jnp.eye(2, dtype=jnp.complex64)
h = -jnp.tensordot(sz, sz, axes=0).transpose((0, 2, 1, 3)).reshape((4, 4)) -\
0.5 * magnetic_field *\
(jnp.tensordot(sx, eye, axes=0).transpose((0, 2, 1, 3)).reshape((4, 4)) +\
  jnp.tensordot(eye, sx, axes=0).transpose((0, 2, 1, 3)).reshape((4, 4)))

# initial random set of parameters
key = random.PRNGKey(42)
key, subkey = random.split(key)
params = random.normal(subkey, (2 * layers_number,))

# loss function
def loss(params: jnp.ndarray) -> jnp.ndarray:
  gates = params2gates(params, qubits_number)
  dens_matrices = fwd_circ(gates, [])
  return energy(dens_matrices, h)

# value and gradient
_loss_val_and_grad = value_and_grad(loss)

# since we use scipy.optimize one needs to convert
# output value and gradient to np.array
def loss_val_and_grad(params):
  v, g = _loss_val_and_grad(params)
  return np.array(v, dtype=np.float64), np.array(g, dtype=np.float64)

# finally we run L-BFGD-B optimozation method
start = time.time()
result = minimize(
  loss_val_and_grad,
  np.asarray(params),
  method = 'L-BFGS-B',
  jac = True,
  options={'maxiter': max_iters_number, 'disp': True},
)
end = time.time()
e = result.fun
# exact ground state energy for magnetic_field = 1 (the phase transition point)
exact_e = -2 * (1 / np.sin(np.pi / (2 * qubits_number)))
evals_num = result.nfev
print('Exact energy: {}'.format(exact_e))
print('Found energy: {}'.format(e))
print('Relative error: {}'.format(np.abs(e - exact_e) / np.abs(exact_e)))
print('Number of loss value and gradient calls: {}'.format(evals_num))
print('Time per loss value and gradient call: {}'.format((end - start) / evals_num))
