#include <cuComplex.h>
#include <cstdint>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

//TODO: adjust kernel parameters
#define BLOCKS_NUM 128
#define THREADS_NUM 128 // must be 2^n

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

__constant__ cuFloatComplex q1g[4];
__constant__ cuFloatComplex q2g[16];

// utility function necessary to traverse a state
__device__ size_t insert_zero(
  size_t mask,
  size_t offset
)
{
  return ((mask & offset) << 1) | ((~mask) & offset);
}

// allocate an uninitialized state on the device
extern "C"
int32_t get_state (
  cuFloatComplex** state,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  int32_t status = cudaMalloc(state, size * sizeof(cuFloatComplex));
  return status;
}

// copy a state to the host
extern "C"
int32_t copy_to_host (
  cuFloatComplex* state,
  cuFloatComplex* host_state,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  int32_t status = cudaMemcpy(host_state, state, size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
  return status;
}

// drop state on the device
extern "C"
int32_t drop_state(
  cuFloatComplex* state
)
{
  int32_t status = cudaFree(state);
  return status;
}

// initialize state to the standard one
__global__ void _set2standard (
  cuFloatComplex* state,
  size_t  qubits_number
)
{
  for (
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    tid < (1 << qubits_number);
    tid += blockDim.x * gridDim.x
  )
  {
    state[tid] = {0., 0.};
  }
  __syncthreads();
  if ( blockIdx.x == 0 && threadIdx.x == 0 ) {
    state[0] = {1., 0.};
  }
}

extern "C"
void set2standard (
  cuFloatComplex* state,
  size_t  qubits_number
)
{
  _set2standard<<<BLOCKS_NUM, THREADS_NUM>>> (
    state,
    qubits_number
  );
}

// computes q1 gate gradient from bwd and fwd "states"
__global__ void _q1grad (
  const cuFloatComplex* fwd,
  const cuFloatComplex* bwd,
  cuFloatComplex* grad,
  size_t pos,
  size_t qubits_number
)
{
  __shared__ cuFloatComplex cache[4 * THREADS_NUM];
  size_t mask =  SIZE_MAX << pos;
  size_t stride = 1 << pos;
  size_t size = 1 << qubits_number;
  size_t batch_size = size >> 1;
  cuFloatComplex tmp[4] = { 
    {0., 0.}, {0., 0.},
    {0., 0.}, {0., 0.},
  };
  for (
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    tid < batch_size;
    tid += blockDim.x * gridDim.x
  )
  {
    size_t btid = insert_zero(mask, tid);
    for (int q = 0; q < 2; q++) {
      for (int p = 0; p < 2; p++) {
        tmp[2 * p + q] = cuCaddf(
          tmp[2 * p + q],
          cuCmulf(
            bwd[p * stride + btid],
            fwd[q * stride + btid]
          )
        );
      }
    }
  }
  for (int q = 0; q < 2; q++) {
    for (int p = 0; p < 2; p++) {
      cache[2 * p + q + 4 * threadIdx.x] = tmp[2 * p + q];
    }
  }
  __syncthreads();
  int s = THREADS_NUM / 2;
  while ( s != 0 ) {
    if ( threadIdx.x < s ) {
      for (int q = 0; q < 2; q++) {
        for (int p = 0; p < 2; p++) {
          cache[2 * p + q + 4 * threadIdx.x] = cuCaddf(
            cache[2 * p + q + 4 * threadIdx.x],
            cache[2 * p + q + 4 * (threadIdx.x + s)]
          );
        }
      }
    }
    __syncthreads();
    s /= 2;
  }
  if (threadIdx.x == 0) {
    for (int q = 0; q < 2; q++) {
      for (int p = 0; p < 2; p++) {
        grad[2 * p + q + 4 * blockIdx.x] = cache[2 * p + q];
      }
    }
  }
}

extern "C"
int q1grad (
  const cuFloatComplex* fwd,
  const cuFloatComplex* bwd,
  cuFloatComplex* grad,
  size_t pos,
  size_t qubits_number
)
{
  cuFloatComplex* device_grad;
  cuFloatComplex* host_grad;
  int32_t alloc_status = cudaMalloc(&device_grad, 4 * BLOCKS_NUM * sizeof(cuFloatComplex));
  _q1grad<<<BLOCKS_NUM, THREADS_NUM>>>(
    fwd,
    bwd,
    device_grad,
    pos,
    qubits_number
  );
  host_grad = (cuFloatComplex*)malloc(4 * BLOCKS_NUM * sizeof(cuFloatComplex));
  int32_t memcopy_status = cudaMemcpy(
    host_grad,
    device_grad,
    4 * BLOCKS_NUM * sizeof(cuFloatComplex),
    cudaMemcpyDeviceToHost
  );
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    for (int q = 0; q < 2; q++) {
      for (int p = 0; p < 2; p++) {
        grad[2 * p + q] = cuCaddf(
          grad[2 * p + q],
          host_grad[2 * p + q + 4 * i]
        );
      }
    }
  }
  delete[] host_grad;
  int32_t free_status = cudaFree(device_grad);
  // return the first error code
  if ( alloc_status != 0 ) return alloc_status;
  if ( memcopy_status != 0 ) return memcopy_status;
  if ( free_status != 0 ) return free_status;
  return 0;
}

// computes q2 hate gradient from fwd and bwd "states"
__global__ void _q2grad (
  const cuFloatComplex* fwd,
  const cuFloatComplex* bwd,
  cuFloatComplex* grad,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  __shared__ cuFloatComplex cache[16 * THREADS_NUM];
  size_t min_pos = MIN(pos1, pos2);
  size_t max_pos = MAX(pos1, pos2);
  size_t min_mask =  SIZE_MAX << min_pos;
  size_t max_mask =  SIZE_MAX << max_pos;
  size_t size = 1 << qubits_number;
  size_t stride1 = 1 << pos1;
  size_t stride2 = 1 << pos2;
  size_t batch_size = size >> 2;
  cuFloatComplex tmp[16] = { 
    {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.},
    {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.},
    {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.},
    {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.},
  };
  for (
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    tid < batch_size;
    tid += blockDim.x * gridDim.x
  )
  {
    size_t btid = insert_zero(max_mask, insert_zero(min_mask, tid));
    for (int q1 = 0; q1 < 2; q1++) {
      for (int q2 = 0; q2 < 2; q2++) {
        for (int p1 = 0; p1 < 2; p1++) {
          for (int p2 = 0; p2 < 2; p2++) {
            tmp[8 * p2 + 4 * p1 + 2 * q2 + q1] = cuCaddf(
              tmp[8 * p2 + 4 * p1 + 2 * q2 + q1],
              cuCmulf(
                bwd[p2 * stride2 + p1 * stride1 + btid],
                fwd[q2 * stride2 + q1 * stride1 + btid]
              )
            );
          }
        }
      }
    }
  }
  for (int q1 = 0; q1 < 2; q1++) {
    for (int q2 = 0; q2 < 2; q2++) {
      for (int p1 = 0; p1 < 2; p1++) {
        for (int p2 = 0; p2 < 2; p2++) {
          cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * threadIdx.x] = tmp[8 * p2 + 4 * p1 + 2 * q2 + q1];
        }
      }
    }
  }
  __syncthreads();
  int s = THREADS_NUM / 2;
  while ( s != 0 ) {
    if ( threadIdx.x < s ) {
      for (int q1 = 0; q1 < 2; q1++) {
        for (int q2 = 0; q2 < 2; q2++) {
          for (int p1 = 0; p1 < 2; p1++) {
            for (int p2 = 0; p2 < 2; p2++) {
              cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * threadIdx.x] = cuCaddf(
                cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * threadIdx.x],
                cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * (threadIdx.x + s)]
              );
            }
          }
        }
      }
    }
    __syncthreads();
    s /= 2;
  }
  if (threadIdx.x == 0) {
    for (int q1 = 0; q1 < 2; q1++) {
      for (int q2 = 0; q2 < 2; q2++) {
        for (int p1 = 0; p1 < 2; p1++) {
          for (int p2 = 0; p2 < 2; p2++) {
            grad[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * blockIdx.x] = cache[8 * p2 + 4 * p1 + 2 * q2 + q1];
          }
        }
      }
    }
  }
}

extern "C"
int q2grad (
  const cuFloatComplex* fwd,
  const cuFloatComplex* bwd,
  cuFloatComplex* grad,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  cuFloatComplex* device_grad;
  cuFloatComplex* host_grad;
  int32_t alloc_status = cudaMalloc(&device_grad, 16 * BLOCKS_NUM * sizeof(cuFloatComplex));
  _q2grad<<<BLOCKS_NUM, THREADS_NUM>>>(
    fwd,
    bwd,
    device_grad,
    pos2,
    pos1,
    qubits_number
  );
  host_grad = (cuFloatComplex*)malloc(16 * BLOCKS_NUM * sizeof(cuFloatComplex));
  int32_t memcopy_status = cudaMemcpy(
    host_grad,
    device_grad,
    16 * BLOCKS_NUM * sizeof(cuFloatComplex),
    cudaMemcpyDeviceToHost
  );
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    for (int q1 = 0; q1 < 2; q1++) {
      for (int q2 = 0; q2 < 2; q2++) {
        for (int p1 = 0; p1 < 2; p1++) {
          for (int p2 = 0; p2 < 2; p2++) {
            grad[8 * p2 + 4 * p1 + 2 * q2 + q1] = cuCaddf(
              grad[8 * p2 + 4 * p1 + 2 * q2 + q1],
              host_grad[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * i]
            );
          }
        }
      }
    }
  }
  delete[] host_grad;
  int32_t free_status = cudaFree(device_grad);
  // return the first error code
  if ( alloc_status != 0 ) return alloc_status;
  if ( memcopy_status != 0 ) return memcopy_status;
  if ( free_status != 0 ) return free_status;
  return 0;
}

// sets state from host
extern "C"
int32_t set_from_host (
  cuFloatComplex* device_state,
  const cuFloatComplex* host_state,
  size_t qubits_number
)
{
  int32_t memcpy_state = cudaMemcpy(device_state, host_state, (1 << qubits_number) * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
  return memcpy_state;
}

// one qubits gate application
__global__ void _q1gate(
  cuFloatComplex* state,
  size_t pos,
  size_t qubits_number
)
{
  size_t mask =  SIZE_MAX << pos;
  size_t stride = 1 << pos;
  size_t size = 1 << qubits_number;
  size_t batch_size = size >> 1;
  for (
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    tid < batch_size;
    tid += blockDim.x * gridDim.x
  )
  {
    size_t btid = insert_zero(mask, tid);
    cuFloatComplex tmp = cuCaddf(
      cuCmulf(q1g[0], state[btid]),
      cuCmulf(q1g[1], state[btid + stride])
    );
    state[stride + btid] = cuCaddf(
      cuCmulf(q1g[2], state[btid]),
      cuCmulf(q1g[3], state[btid + stride])
    );
    state[btid] = tmp;
  }
}

extern "C"
int32_t q1gate(
  cuFloatComplex* state,
  const cuFloatComplex* gate,
  size_t idx,
  size_t qubits_number
)
{
  int32_t copy_status = cudaMemcpyToSymbol(q1g, gate, 4 * sizeof(cuFloatComplex));
  _q1gate<<<BLOCKS_NUM, THREADS_NUM>>>(state, idx, qubits_number);
  return copy_status;
}

// two qubits gate application
__global__ void _q2gate(
  cuFloatComplex* state,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  size_t min_pos = MIN(pos1, pos2);
  size_t max_pos = MAX(pos1, pos2);
  size_t min_mask =  SIZE_MAX << min_pos;
  size_t max_mask =  SIZE_MAX << max_pos;
  size_t size = 1 << qubits_number;
  size_t stride1 = 1 << pos1;
  size_t stride2 = 1 << pos2;
  size_t batch_size = size >> 2;
  for (
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    tid < batch_size;
    tid += blockDim.x * gridDim.x
  )
  {
    size_t btid = insert_zero(max_mask, insert_zero(min_mask, tid));
    cuFloatComplex tmp[4] = { {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.} };
    for (size_t p1 = 0; p1 < 2; p1++) {
      for (size_t p2 = 0; p2 < 2; p2++) {
        for (size_t q1 = 0; q1 < 2; q1++) {
          for (size_t q2 = 0; q2 < 2; q2++) {
            tmp[2 * q2 + q1] = cuCaddf(
              tmp[2 * q2 + q1],
              cuCmulf(
                q2g[8 * q2 + 4 * q1 + 2 * p2 + p1],
                state[stride2 * p2 + stride1 * p1 + btid]
              )
            );
          }
        }
      }
    }
    state[btid] = tmp[0];
    state[btid + stride1] = tmp[1];
    state[btid + stride2] = tmp[2];
    state[btid + stride1 + stride2] = tmp[3];
  }
}

extern "C"
int32_t q2gate(
  cuFloatComplex* state,
  const cuFloatComplex* gate,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  int32_t copy_status = cudaMemcpyToSymbol(q2g, gate, 16 * sizeof(cuFloatComplex));
  _q2gate<<<BLOCKS_NUM, THREADS_NUM>>>(state, pos2, pos1, qubits_number);
  return copy_status;
}

// one qubit density matrix computation
__global__ void _get_q1density(
  const cuFloatComplex* state,
  cuFloatComplex* density,
  size_t pos,
  size_t qubits_number
)
{
  __shared__ cuFloatComplex cache[4 * THREADS_NUM];
  size_t mask =  SIZE_MAX << pos;
  size_t stride = 1 << pos;
  size_t size = 1 << qubits_number;
  size_t batch_size = size >> 1;
  cuFloatComplex tmp[4] = { 
    {0., 0.}, {0., 0.},
    {0., 0.}, {0., 0.},
  };
  for (
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    tid < batch_size;
    tid += blockDim.x * gridDim.x
  )
  {
    size_t btid = insert_zero(mask, tid);
    for (int q = 0; q < 2; q++) {
      for (int p = 0; p < 2; p++) {
        tmp[2 * p + q] = cuCaddf(
          tmp[2 * p + q],
          cuCmulf(
            state[p * stride + btid],
            cuConjf(state[q * stride + btid])
          )
        );
      }
    }
  }
  for (int q = 0; q < 2; q++) {
    for (int p = 0; p < 2; p++) {
      cache[2 * p + q + 4 * threadIdx.x] = tmp[2 * p + q];
    }
  }
  __syncthreads();
  int s = THREADS_NUM / 2;
  while ( s != 0 ) {
    if ( threadIdx.x < s ) {
      for (int q = 0; q < 2; q++) {
        for (int p = 0; p < 2; p++) {
          cache[2 * p + q + 4 * threadIdx.x] = cuCaddf(
            cache[2 * p + q + 4 * threadIdx.x],
            cache[2 * p + q + 4 * (threadIdx.x + s)]
          );
        }
      }
    }
    __syncthreads();
    s /= 2;
  }
  if (threadIdx.x == 0) {
    for (int q = 0; q < 2; q++) {
      for (int p = 0; p < 2; p++) {
        density[2 * p + q + 4 * blockIdx.x] = cache[2 * p + q];
      }
    }
  }
}

extern "C"
int32_t get_q1density(
  const cuFloatComplex* state,
  cuFloatComplex* density,
  size_t pos,
  size_t qubits_number
)
{
  cuFloatComplex* device_density;
  cuFloatComplex* host_density;
  int32_t alloc_status = cudaMalloc(&device_density, 4 * BLOCKS_NUM * sizeof(cuFloatComplex));
  _get_q1density<<<BLOCKS_NUM, THREADS_NUM>>>(
    state,
    device_density,
    pos,
    qubits_number
  );
  host_density = (cuFloatComplex*)malloc(4 * BLOCKS_NUM * sizeof(cuFloatComplex));
  int32_t memcopy_status = cudaMemcpy(
    host_density,
    device_density,
    4 * BLOCKS_NUM * sizeof(cuFloatComplex),
    cudaMemcpyDeviceToHost
  );
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    for (int q = 0; q < 2; q++) {
      for (int p = 0; p < 2; p++) {
        density[2 * p + q] = cuCaddf(
          density[2 * p + q],
          host_density[2 * p + q + 4 * i]
        );
      }
    }
  }
  delete[] host_density;
  int32_t free_status = cudaFree(device_density);
  // return the first error code
  if ( alloc_status != 0 ) return alloc_status;
  if ( memcopy_status != 0 ) return memcopy_status;
  if ( free_status != 0 ) return free_status;
  return 0;
}

// two qubit density matrix computation
__global__ void _get_q2density(
  const cuFloatComplex* state,
  cuFloatComplex* density,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  __shared__ cuFloatComplex cache[16 * THREADS_NUM];
  size_t min_pos = MIN(pos1, pos2);
  size_t max_pos = MAX(pos1, pos2);
  size_t min_mask =  SIZE_MAX << min_pos;
  size_t max_mask =  SIZE_MAX << max_pos;
  size_t size = 1 << qubits_number;
  size_t stride1 = 1 << pos1;
  size_t stride2 = 1 << pos2;
  size_t batch_size = size >> 2;
  cuFloatComplex tmp[16] = {
    {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.},
    {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.},
    {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.},
    {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.},
  };
  for (
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    tid < batch_size;
    tid += blockDim.x * gridDim.x
  )
  {
    size_t btid = insert_zero(max_mask, insert_zero(min_mask, tid));
    for (int q1 = 0; q1 < 2; q1++) {
      for (int q2 = 0; q2 < 2; q2++) {
        for (int p1 = 0; p1 < 2; p1++) {
          for (int p2 = 0; p2 < 2; p2++) {
            tmp[8 * p2 + 4 * p1 + 2 * q2 + q1] = cuCaddf(
              tmp[8 * p2 + 4 * p1 + 2 * q2 + q1],
              cuCmulf(
                state[p2 * stride2 + p1 * stride1 + btid],
                cuConjf(state[q2 * stride2 + q1 * stride1 + btid])
              )
            );
          }
        }
      }
    }
  }
  for (int q1 = 0; q1 < 2; q1++) {
    for (int q2 = 0; q2 < 2; q2++) {
      for (int p1 = 0; p1 < 2; p1++) {
        for (int p2 = 0; p2 < 2; p2++) {
          cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * threadIdx.x] = tmp[8 * p2 + 4 * p1 + 2 * q2 + q1];
        }
      }
    }
  }
  __syncthreads();
  int s = THREADS_NUM / 2;
  while ( s != 0 ) {
    if ( threadIdx.x < s ) {
      for (int q1 = 0; q1 < 2; q1++) {
        for (int q2 = 0; q2 < 2; q2++) {
          for (int p1 = 0; p1 < 2; p1++) {
            for (int p2 = 0; p2 < 2; p2++) {
              cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * threadIdx.x] = cuCaddf(
                cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * threadIdx.x],
                cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * (threadIdx.x + s)]
              );
            }
          }
        }
      }
    }
    __syncthreads();
    s /= 2;
  }
  if (threadIdx.x == 0) {
    for (int q1 = 0; q1 < 2; q1++) {
      for (int q2 = 0; q2 < 2; q2++) {
        for (int p1 = 0; p1 < 2; p1++) {
          for (int p2 = 0; p2 < 2; p2++) {
            density[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * blockIdx.x] = cache[8 * p2 + 4 * p1 + 2 * q2 + q1];
          }
        }
      }
    }
  }
}

extern "C"
int32_t get_q2density(
  const cuFloatComplex* state,
  cuFloatComplex* density,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  cuFloatComplex* device_density;
  cuFloatComplex* host_density;
  int32_t alloc_status = cudaMalloc(&device_density, 16 * BLOCKS_NUM * sizeof(cuFloatComplex));
  _get_q2density<<<BLOCKS_NUM, THREADS_NUM>>>(
    state,
    device_density,
    pos2,
    pos1,
    qubits_number
  );
  host_density = (cuFloatComplex*)malloc(16 * BLOCKS_NUM * sizeof(cuFloatComplex));
  int32_t memcopy_status = cudaMemcpy(
    host_density,
    device_density,
    16 * BLOCKS_NUM * sizeof(cuFloatComplex),
    cudaMemcpyDeviceToHost
  );
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    for (int q1 = 0; q1 < 2; q1++) {
      for (int q2 = 0; q2 < 2; q2++) {
        for (int p1 = 0; p1 < 2; p1++) {
          for (int p2 = 0; p2 < 2; p2++) {
            density[8 * p2 + 4 * p1 + 2 * q2 + q1] = cuCaddf(
              density[8 * p2 + 4 * p1 + 2 * q2 + q1],
              host_density[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * i]
            );
          }
        }
      }
    }
  }
  delete[] host_density;
  int32_t free_status = cudaFree(device_density);
  // return the first error code
  if ( alloc_status != 0 ) return alloc_status;
  if ( memcopy_status != 0 ) return memcopy_status;
  if ( free_status != 0 ) return free_status;
  return 0;
}

// copy of a state
__global__ void _copy(
  const cuFloatComplex* src,
  cuFloatComplex* dst,
  size_t qubits_number
)
{
  for (
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    tid < (1 << qubits_number);
    tid += blockDim.x * gridDim.x
  )
  {
    dst[tid] = src[tid];
  }
}

extern "C"
void copy(
  const cuFloatComplex* src,
  cuFloatComplex* dst,
  size_t qubits_number
)
{
  _copy<<<BLOCKS_NUM, THREADS_NUM>>>(
    src,
    dst,
    qubits_number
  );
}

// primitives to pass gradient through the density matrix computation 
__global__ void _conj_and_double(
  const cuFloatComplex* src,
  cuFloatComplex* dst,
  size_t qubits_number
)
{
  for (
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    tid < (1 << qubits_number);
    tid += blockDim.x * gridDim.x
  )
  {
    dst[tid].x = 2 * src[tid].x;
    dst[tid].y = -2 * src[tid].y;
  }
}

extern "C"
void conj_and_double(
  const cuFloatComplex* src,
  cuFloatComplex* dst,
  size_t qubits_number
)
{
  _conj_and_double<<<BLOCKS_NUM, THREADS_NUM>>>(
    src,
    dst,
    qubits_number
  );
}

__global__ void _add(
  const cuFloatComplex* src,
  cuFloatComplex* dst,
  size_t qubits_number
)
{
  for (
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    tid < (1 << qubits_number);
    tid += blockDim.x * gridDim.x
  )
  {
    dst[tid] = cuCaddf(dst[tid], src[tid]);
  }
}

extern "C"
void add(
  const cuFloatComplex* src,
  cuFloatComplex* dst,
  size_t qubits_number
)
{
  _add<<<BLOCKS_NUM, THREADS_NUM>>>(
    src,
    dst,
    qubits_number
  );
}
