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

__constant__ cuComplex q1g[4];
__constant__ cuComplex q2g[16];

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
  cuComplex** state,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  int32_t status = cudaMalloc(state, size * sizeof(cuComplex));
  return status;
}

// copy a state to the host
extern "C"
int32_t copy_to_host (
  cuComplex* state,
  cuComplex* host_state,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  int32_t status = cudaMemcpy(host_state, state, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
  return status;
}

// drop state on the device
extern "C"
int32_t drop_state(
  cuComplex* state
)
{
  int32_t status = cudaFree(state);
  return status;
}

// initialize state to the standard one
__global__ void _set2standard (
  cuComplex* state,
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
  cuComplex* state,
  size_t  qubits_number
)
{
  _set2standard<<<BLOCKS_NUM, THREADS_NUM>>> (
    state,
    qubits_number
  );
}

// computes l2 norm of a state
__global__ void _norm (
  const cuComplex* state,
  float* result,
  size_t qubits_number
)
{
  __shared__ float cache[THREADS_NUM];
  float tmp = 0;
  for (
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    tid < (1 << qubits_number);
    tid += blockDim.x * gridDim.x
  )
  {
    float real = state[tid].x;
    float imag = state[tid].y;
    tmp += real * real + imag * imag;
  }
  cache[threadIdx.x] = tmp;
  __syncthreads();
  int s = blockDim.x / 2;
  while ( s != 0 ) {
    if ( threadIdx.x < s ) {
      cache[threadIdx.x] += cache[threadIdx.x + s];
    }
    __syncthreads();
    s /= 2;
  }
  if (threadIdx.x == 0) {
    result[blockIdx.x] = cache[0];
  }
}

// sets state from host
extern "C"
int32_t set_from_host (
  cuComplex* device_state,
  const cuComplex* host_state,
  size_t qubits_number
)
{
  int32_t memcpy_state = cudaMemcpy(device_state, host_state, (1 << qubits_number) * sizeof(cuComplex), cudaMemcpyHostToDevice);
  return memcpy_state;
}

extern "C"
int32_t norm (
  const cuComplex* state,
  float* result,
  size_t qubits_number
)
{
  float* device_result;
  float* host_result;
  host_result = (float*)malloc(BLOCKS_NUM * sizeof(float));
  int32_t alloc_status = cudaMalloc(&device_result, BLOCKS_NUM * sizeof(float));
  _norm<<<BLOCKS_NUM, THREADS_NUM>>>(state, device_result, qubits_number);
  int32_t memcopy_status = cudaMemcpy(
    host_result,
    device_result,
    BLOCKS_NUM * sizeof(float),
    cudaMemcpyDeviceToHost
  );
  for (int i = 0; i < BLOCKS_NUM; i++) {
    *result += host_result[i];
  }
  delete[] host_result;
  int32_t free_status = cudaFree(device_result);
  // return the first error code
  if ( alloc_status != 0 ) return alloc_status;
  if ( memcopy_status != 0 ) return memcopy_status;
  if ( free_status != 0 ) return free_status;
  return 0;
}

// one qubits gate application
__global__ void _q1gate(
  cuComplex* state,
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
    cuComplex tmp = cuCaddf(
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
  cuComplex* state,
  const cuComplex* gate,
  size_t idx,
  size_t qubits_number
)
{
  int32_t copy_status = cudaMemcpyToSymbol(q1g, gate, 4 * sizeof(cuComplex));
  _q1gate<<<BLOCKS_NUM, THREADS_NUM>>>(state, idx, qubits_number);
  return copy_status;
}

// two qubits gate application
__global__ void _q2gate(
  cuComplex* state,
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
    cuComplex tmp[4] = { {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.} };
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
  cuComplex* state,
  const cuComplex* gate,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  int32_t copy_status = cudaMemcpyToSymbol(q2g, gate, 16 * sizeof(cuComplex));
  _q2gate<<<BLOCKS_NUM, THREADS_NUM>>>(state, pos2, pos1, qubits_number);
  return copy_status;
}

// one qubit density matrix computation
__global__ void _get_q1density(
  const cuComplex* state,
  cuComplex* density,
  size_t pos,
  size_t qubits_number
)
{
  __shared__ cuComplex cache[4 * THREADS_NUM];
  size_t mask =  SIZE_MAX << pos;
  size_t stride = 1 << pos;
  size_t size = 1 << qubits_number;
  size_t batch_size = size >> 1;
  cuComplex tmp[4] = { 
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
    s /= 2;
  }
  __syncthreads();
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
  const cuComplex* state,
  cuComplex* density,
  size_t pos,
  size_t qubits_number
)
{
  cuComplex* device_density;
  cuComplex* host_density;
  int32_t alloc_status = cudaMalloc(&device_density, 4 * BLOCKS_NUM * sizeof(cuComplex));
  _get_q1density<<<BLOCKS_NUM, THREADS_NUM>>>(
    state,
    device_density,
    pos,
    qubits_number
  );
  host_density = (cuComplex*)malloc(4 * BLOCKS_NUM * sizeof(cuComplex));
  int32_t memcopy_status = cudaMemcpy(
    host_density,
    device_density,
    4 * BLOCKS_NUM * sizeof(cuComplex),
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
  const cuComplex* state,
  cuComplex* density,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  __shared__ cuComplex cache[16 * THREADS_NUM];
  size_t min_pos = MIN(pos1, pos2);
  size_t max_pos = MAX(pos1, pos2);
  size_t min_mask =  SIZE_MAX << min_pos;
  size_t max_mask =  SIZE_MAX << max_pos;
  size_t size = 1 << qubits_number;
  size_t stride1 = 1 << pos1;
  size_t stride2 = 1 << pos2;
  size_t batch_size = size >> 2;
  cuComplex tmp[16] = {
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
    s /= 2;
  }
  __syncthreads();
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
  const cuComplex* state,
  cuComplex* density,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  cuComplex* device_density;
  cuComplex* host_density;
  int32_t alloc_status = cudaMalloc(&device_density, 16 * BLOCKS_NUM * sizeof(cuComplex));
  _get_q2density<<<BLOCKS_NUM, THREADS_NUM>>>(
    state,
    device_density,
    pos2,
    pos1,
    qubits_number
  );
  host_density = (cuComplex*)malloc(16 * BLOCKS_NUM * sizeof(cuComplex));
  int32_t memcopy_status = cudaMemcpy(
    host_density,
    device_density,
    16 * BLOCKS_NUM * sizeof(cuComplex),
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
