#include <cuComplex.h>
#include <cstdint>

//TODO: adjust kernel parameters
#define BLOCKS_NUM 128
#define THREADS_NUM 128 // must be 2^n

#ifdef F64
  #define ADD cuCadd
  #define SUB cuCsub
  #define MUL cuCmul
  #define CONJ cuConj
  #define COMPLEX cuDoubleComplex
  #define COMPLEXNEW make_cuDoubleComplex
  #define ABS cuCabs
#else
  #define ADD cuCaddf
  #define SUB cuCsubf
  #define MUL cuCmulf
  #define CONJ cuConjf
  #define COMPLEX cuFloatComplex
  #define COMPLEXNEW make_cuFloatComplex
  #define ABS cuCabsf
#endif

// macro definitions of the most used for loops
#define PARALLEL_FOR(index, stop_index, ...)              \
  for (                                                   \
    size_t index = threadIdx.x + blockIdx.x * blockDim.x; \
    index < stop_index;                                   \
    index += blockDim.x * gridDim.x                       \
  )                                                       \
  {                                                       \
    __VA_ARGS__                                           \
  }

#define ONE_POSITION_FOR(...)                             \
  for (int q = 0; q < 2; q++) {                           \
    for (int p = 0; p < 2; p++) {                         \
      __VA_ARGS__                                         \
    }                                                     \
  }

#define TWO_POSITIONS_FOR(...)                            \
  for (int q1 = 0; q1 < 2; q1++) {                        \
    for (int q2 = 0; q2 < 2; q2++) {                      \
      for (int p1 = 0; p1 < 2; p1++) {                    \
        for (int p2 = 0; p2 < 2; p2++) {                  \
          __VA_ARGS__                                     \
        }                                                 \
      }                                                   \
    }                                                     \
  }

__constant__ COMPLEX q1g[4];
__constant__ COMPLEX q2g[16];
__constant__ COMPLEX q2dg[4];

// utility functions necessary to traverse a state
__device__ size_t insert_zero(
  size_t mask,
  size_t offset
)
{
  return ((mask & offset) << 1) | ((~mask) & offset);
}

__device__ size_t insert_two_zeros(
  size_t mask1,
  size_t mask2,
  size_t offset
)
{
  size_t min_index_mask = mask1 > mask2 ? mask1 : mask2;
  size_t max_index_mask = mask1 > mask2 ? mask2 : mask1;
  return insert_zero(max_index_mask, insert_zero(min_index_mask, offset));
}

// allocate an uninitialized state on the device
extern "C"
int32_t get_state (
  COMPLEX** state,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  int32_t status = cudaMalloc(state, size * sizeof(COMPLEX));
  return status;
}

// copy a state to the host
extern "C"
int32_t copy_to_host (
  COMPLEX* state,
  COMPLEX* host_state,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  int32_t status = cudaMemcpy(host_state, state, size * sizeof(COMPLEX), cudaMemcpyDeviceToHost);
  return status;
}

// drop state on the device
extern "C"
int32_t drop_state(
  COMPLEX* state
)
{
  int32_t status = cudaFree(state);
  return status;
}

// initialize state to the standard one
__global__ void _set2standard (
  COMPLEX* state,
  size_t  qubits_number
)
{
  size_t size = 1 << qubits_number;
  PARALLEL_FOR(tid, size, state[tid] = {0, 0};)
  __syncthreads();
  if ( blockIdx.x == 0 && threadIdx.x == 0 ) {
    state[0] = {1, 0};
  }
}

extern "C"
void set2standard (
  COMPLEX* state,
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
  const COMPLEX* fwd,
  const COMPLEX* bwd,
  COMPLEX* grad,
  size_t pos,
  size_t qubits_number
)
{
  __shared__ COMPLEX cache[4 * THREADS_NUM];
  size_t mask =  SIZE_MAX << pos;
  size_t stride = 1 << pos;
  size_t size = 1 << qubits_number;
  size_t batch_size = size >> 1;
  COMPLEX tmp[4] = { 
    {0, 0}, {0, 0},
    {0, 0}, {0, 0},
  };
  PARALLEL_FOR(tid, batch_size,
    size_t btid = insert_zero(mask, tid);
    ONE_POSITION_FOR(
      tmp[2 * p + q] = ADD(
        tmp[2 * p + q],
        MUL(
          bwd[p * stride + btid],
          fwd[q * stride + btid]
        )
      );
    )
  )
  ONE_POSITION_FOR(
    cache[2 * p + q + 4 * threadIdx.x] = tmp[2 * p + q];
  )
  __syncthreads();
  int s = THREADS_NUM / 2;
  while ( s != 0 ) {
    if ( threadIdx.x < s ) {
      ONE_POSITION_FOR(
        cache[2 * p + q + 4 * threadIdx.x] = ADD(
          cache[2 * p + q + 4 * threadIdx.x],
          cache[2 * p + q + 4 * (threadIdx.x + s)]
        );
      )
    }
    __syncthreads();
    s /= 2;
  }
  if (threadIdx.x == 0) {
    ONE_POSITION_FOR(
      grad[2 * p + q + 4 * blockIdx.x] = cache[2 * p + q];
    )
  }
}

extern "C"
int q1grad (
  const COMPLEX* fwd,
  const COMPLEX* bwd,
  COMPLEX* grad,
  size_t pos,
  size_t qubits_number
)
{
  COMPLEX* device_grad;
  COMPLEX* host_grad;
  int32_t alloc_status = cudaMalloc(&device_grad, 4 * BLOCKS_NUM * sizeof(COMPLEX));
  _q1grad<<<BLOCKS_NUM, THREADS_NUM>>>(
    fwd,
    bwd,
    device_grad,
    pos,
    qubits_number
  );
  host_grad = (COMPLEX*)malloc(4 * BLOCKS_NUM * sizeof(COMPLEX));
  int32_t memcopy_status = cudaMemcpy(
    host_grad,
    device_grad,
    4 * BLOCKS_NUM * sizeof(COMPLEX),
    cudaMemcpyDeviceToHost
  );
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    ONE_POSITION_FOR(
      grad[2 * p + q] = ADD(
        grad[2 * p + q],
        host_grad[2 * p + q + 4 * i]
      );
    )
  }
  delete[] host_grad;
  int32_t free_status = cudaFree(device_grad);
  // return the first error code
  if ( alloc_status != 0 ) return alloc_status;
  if ( memcopy_status != 0 ) return memcopy_status;
  if ( free_status != 0 ) return free_status;
  return 0;
}

// computes q2 gate gradient from fwd and bwd "states"
__global__ void _q2grad (
  const COMPLEX* fwd,
  const COMPLEX* bwd,
  COMPLEX* grad,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  __shared__ COMPLEX cache[16 * THREADS_NUM];
  size_t size = 1 << qubits_number;
  size_t mask1 =  SIZE_MAX << pos1;
  size_t mask2 =  SIZE_MAX << pos2;
  size_t stride1 = 1 << pos1;
  size_t stride2 = 1 << pos2;
  size_t batch_size = size >> 2;
  COMPLEX tmp[16] = { 
    {0, 0}, {0, 0}, {0, 0}, {0, 0},
    {0, 0}, {0, 0}, {0, 0}, {0, 0},
    {0, 0}, {0, 0}, {0, 0}, {0, 0},
    {0, 0}, {0, 0}, {0, 0}, {0, 0},
  };
  PARALLEL_FOR(tid, batch_size,
    size_t btid = insert_two_zeros(mask1, mask2, tid);
    TWO_POSITIONS_FOR(
      tmp[8 * p2 + 4 * p1 + 2 * q2 + q1] = ADD(
        tmp[8 * p2 + 4 * p1 + 2 * q2 + q1],
        MUL(
          bwd[p2 * stride2 + p1 * stride1 + btid],
          fwd[q2 * stride2 + q1 * stride1 + btid]
        )
      );
    )
  )
  TWO_POSITIONS_FOR(
    cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * threadIdx.x] = tmp[8 * p2 + 4 * p1 + 2 * q2 + q1];
  )
  __syncthreads();
  int s = THREADS_NUM / 2;
  while ( s != 0 ) {
    if ( threadIdx.x < s ) {
      TWO_POSITIONS_FOR(
        cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * threadIdx.x] = ADD(
          cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * threadIdx.x],
          cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * (threadIdx.x + s)]
        );
      )
    }
    __syncthreads();
    s /= 2;
  }
  if (threadIdx.x == 0) {
    TWO_POSITIONS_FOR(
      grad[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * blockIdx.x] = cache[8 * p2 + 4 * p1 + 2 * q2 + q1];
    )
  }
}

extern "C"
int q2grad (
  const COMPLEX* fwd,
  const COMPLEX* bwd,
  COMPLEX* grad,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  COMPLEX* device_grad;
  COMPLEX* host_grad;
  int32_t alloc_status = cudaMalloc(&device_grad, 16 * BLOCKS_NUM * sizeof(COMPLEX));
  _q2grad<<<BLOCKS_NUM, THREADS_NUM>>>(
    fwd,
    bwd,
    device_grad,
    pos2,
    pos1,
    qubits_number
  );
  host_grad = (COMPLEX*)malloc(16 * BLOCKS_NUM * sizeof(COMPLEX));
  int32_t memcopy_status = cudaMemcpy(
    host_grad,
    device_grad,
    16 * BLOCKS_NUM * sizeof(COMPLEX),
    cudaMemcpyDeviceToHost
  );
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    TWO_POSITIONS_FOR(
      grad[8 * p2 + 4 * p1 + 2 * q2 + q1] = ADD(
        grad[8 * p2 + 4 * p1 + 2 * q2 + q1],
        host_grad[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * i]
      );
    )
  }
  delete[] host_grad;
  int32_t free_status = cudaFree(device_grad);
  // return the first error code
  if ( alloc_status != 0 ) return alloc_status;
  if ( memcopy_status != 0 ) return memcopy_status;
  if ( free_status != 0 ) return free_status;
  return 0;
}

// computes q2 diagonal gate gradient from fwd and bwd "states"
__global__ void _q2grad_diag (
  const COMPLEX* fwd,
  const COMPLEX* bwd,
  COMPLEX* grad,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  __shared__ COMPLEX cache[4 * THREADS_NUM];
  size_t size = 1 << qubits_number;
  size_t mask1 =  SIZE_MAX << pos1;
  size_t mask2 =  SIZE_MAX << pos2;
  size_t stride1 = 1 << pos1;
  size_t stride2 = 1 << pos2;
  size_t batch_size = size >> 2;
  COMPLEX tmp[4] = { {0, 0}, {0, 0}, {0, 0}, {0, 0} };
  PARALLEL_FOR(tid, batch_size,
    size_t btid = insert_two_zeros(mask1, mask2, tid);
    ONE_POSITION_FOR(
      tmp[2 * p + q] = ADD(
        tmp[2 * p + q],
        MUL(
          bwd[p * stride2 + q * stride1 + btid],
          fwd[p * stride2 + q * stride1 + btid]
        )
      );
    )
  )
  ONE_POSITION_FOR(
    cache[2 * p + q + 4 * threadIdx.x] = tmp[2 * p + q];
  )
  __syncthreads();
  int s = THREADS_NUM / 2;
  while ( s != 0 ) {
    if ( threadIdx.x < s ) {
      ONE_POSITION_FOR(
        cache[2 * p + q + 4 * threadIdx.x] = ADD(
          cache[2 * p + q + 4 * threadIdx.x],
          cache[2 * p + q + 4 * (threadIdx.x + s)]
        );
      )
    }
    __syncthreads();
    s /= 2;
  }
  if (threadIdx.x == 0) {
    ONE_POSITION_FOR(
      grad[2 * p + q + 4 * blockIdx.x] = cache[2 * p + q];
    )
  }
}

extern "C"
int q2grad_diag (
  const COMPLEX* fwd,
  const COMPLEX* bwd,
  COMPLEX* grad,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  COMPLEX* device_grad;
  COMPLEX* host_grad;
  int32_t alloc_status = cudaMalloc(&device_grad, 4 * BLOCKS_NUM * sizeof(COMPLEX));
  _q2grad_diag<<<BLOCKS_NUM, THREADS_NUM>>>(
    fwd,
    bwd,
    device_grad,
    pos2,
    pos1,
    qubits_number
  );
  host_grad = (COMPLEX*)malloc(4 * BLOCKS_NUM * sizeof(COMPLEX));
  int32_t memcopy_status = cudaMemcpy(
    host_grad,
    device_grad,
    4 * BLOCKS_NUM * sizeof(COMPLEX),
    cudaMemcpyDeviceToHost
  );
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    ONE_POSITION_FOR(
      grad[2 * p + q] = ADD(
        grad[2 * p + q],
        host_grad[2 * p + q + 4 * i]
      );
    )
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
  COMPLEX* device_state,
  const COMPLEX* host_state,
  size_t qubits_number
)
{
  int32_t memcpy_state = cudaMemcpy(device_state, host_state, (1 << qubits_number) * sizeof(COMPLEX), cudaMemcpyHostToDevice);
  return memcpy_state;
}

// one qubits gate application
__global__ void _q1gate(
  COMPLEX* state,
  size_t pos,
  size_t qubits_number
)
{
  size_t mask =  SIZE_MAX << pos;
  size_t stride = 1 << pos;
  size_t size = 1 << qubits_number;
  size_t batch_size = size >> 1;
  PARALLEL_FOR(tid, batch_size,
    size_t btid = insert_zero(mask, tid);
    COMPLEX tmp[2] = { {0, 0}, {0, 0} };
    ONE_POSITION_FOR(
      tmp[p] = ADD(tmp[p], MUL(q1g[2 * p + q], state[stride * q + btid]));
    )
    state[btid] = tmp[0];
    state[btid + stride] = tmp[1];
  )
}

extern "C"
int32_t q1gate(
  COMPLEX* state,
  const COMPLEX* gate,
  size_t idx,
  size_t qubits_number
)
{
  int32_t copy_status = cudaMemcpyToSymbol(q1g, gate, 4 * sizeof(COMPLEX));
  _q1gate<<<BLOCKS_NUM, THREADS_NUM>>>(state, idx, qubits_number);
  return copy_status;
}

// two qubits gate application
__global__ void _q2gate(
  COMPLEX* state,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  size_t mask1 =  SIZE_MAX << pos1;
  size_t mask2 =  SIZE_MAX << pos2;
  size_t stride1 = 1 << pos1;
  size_t stride2 = 1 << pos2;
  size_t batch_size = size >> 2;
  PARALLEL_FOR(tid, batch_size,
    size_t btid = insert_two_zeros(mask1, mask2, tid);
    COMPLEX tmp[4] = { {0, 0}, {0, 0}, {0, 0}, {0, 0} };
    TWO_POSITIONS_FOR(
      tmp[2 * q2 + q1] = ADD(
        tmp[2 * q2 + q1],
        MUL(
          q2g[8 * q2 + 4 * q1 + 2 * p2 + p1],
          state[stride2 * p2 + stride1 * p1 + btid]
        )
      );
    )
    state[btid] = tmp[0];
    state[btid + stride1] = tmp[1];
    state[btid + stride2] = tmp[2];
    state[btid + stride1 + stride2] = tmp[3];
  )
}

extern "C"
int32_t q2gate(
  COMPLEX* state,
  const COMPLEX* gate,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  int32_t copy_status = cudaMemcpyToSymbol(q2g, gate, 16 * sizeof(COMPLEX));
  _q2gate<<<BLOCKS_NUM, THREADS_NUM>>>(state, pos2, pos1, qubits_number);
  return copy_status;
}

// two qubits diagonal gate application
__global__ void _q2gate_diag(
  COMPLEX* state,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  size_t mask1 =  SIZE_MAX << pos1;
  size_t mask2 =  SIZE_MAX << pos2;
  size_t stride1 = 1 << pos1;
  size_t stride2 = 1 << pos2;
  size_t batch_size = size >> 2;
  PARALLEL_FOR(tid, batch_size,
    size_t btid = insert_two_zeros(mask1, mask2, tid);
    state[btid] = MUL(q2dg[0], state[btid]);
    state[btid + stride1] = MUL(q2dg[1], state[btid + stride1]);
    state[btid + stride2] = MUL(q2dg[2], state[btid + stride2]);
    state[btid + stride1 + stride2] = MUL(q2dg[3], state[btid + stride1 + stride2]);
  )
}

extern "C"
int32_t q2gate_diag(
  COMPLEX* state,
  const COMPLEX* gate,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  int32_t copy_status = cudaMemcpyToSymbol(q2dg, gate, 4 * sizeof(COMPLEX));
  _q2gate_diag<<<BLOCKS_NUM, THREADS_NUM>>>(state, pos2, pos1, qubits_number);
  return copy_status;
}

// one qubit density matrix computation
__global__ void _get_q1density(
  const COMPLEX* state,
  COMPLEX* density,
  size_t pos,
  size_t qubits_number
)
{
  __shared__ COMPLEX cache[4 * THREADS_NUM];
  size_t mask =  SIZE_MAX << pos;
  size_t stride = 1 << pos;
  size_t size = 1 << qubits_number;
  size_t batch_size = size >> 1;
  COMPLEX tmp[4] = { 
    {0, 0}, {0, 0},
    {0, 0}, {0, 0},
  };
  PARALLEL_FOR(tid, batch_size,
    size_t btid = insert_zero(mask, tid);
    ONE_POSITION_FOR(
      tmp[2 * p + q] = ADD(
        tmp[2 * p + q],
        MUL(
          state[p * stride + btid],
          CONJ(state[q * stride + btid])
        )
      );
    )
  )
  ONE_POSITION_FOR(
    cache[2 * p + q + 4 * threadIdx.x] = tmp[2 * p + q];
  )
  __syncthreads();
  int s = THREADS_NUM / 2;
  while ( s != 0 ) {
    if ( threadIdx.x < s ) {
      ONE_POSITION_FOR(
        cache[2 * p + q + 4 * threadIdx.x] = ADD(
          cache[2 * p + q + 4 * threadIdx.x],
          cache[2 * p + q + 4 * (threadIdx.x + s)]
        );
      )
    }
    __syncthreads();
    s /= 2;
  }
  if (threadIdx.x == 0) {
    ONE_POSITION_FOR(
      density[2 * p + q + 4 * blockIdx.x] = cache[2 * p + q];
    )
  }
}

extern "C"
int32_t get_q1density(
  const COMPLEX* state,
  COMPLEX* density,
  size_t pos,
  size_t qubits_number
)
{
  COMPLEX* device_density;
  COMPLEX* host_density;
  int32_t alloc_status = cudaMalloc(&device_density, 4 * BLOCKS_NUM * sizeof(COMPLEX));
  _get_q1density<<<BLOCKS_NUM, THREADS_NUM>>>(
    state,
    device_density,
    pos,
    qubits_number
  );
  host_density = (COMPLEX*)malloc(4 * BLOCKS_NUM * sizeof(COMPLEX));
  int32_t memcopy_status = cudaMemcpy(
    host_density,
    device_density,
    4 * BLOCKS_NUM * sizeof(COMPLEX),
    cudaMemcpyDeviceToHost
  );
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    ONE_POSITION_FOR(
      density[2 * p + q] = ADD(
        density[2 * p + q],
        host_density[2 * p + q + 4 * i]
      );
    )
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
  const COMPLEX* state,
  COMPLEX* density,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  __shared__ COMPLEX cache[16 * THREADS_NUM];
  size_t size = 1 << qubits_number;
  size_t mask1 =  SIZE_MAX << pos1;
  size_t mask2 =  SIZE_MAX << pos2;
  size_t stride1 = 1 << pos1;
  size_t stride2 = 1 << pos2;
  size_t batch_size = size >> 2;
  COMPLEX tmp[16] = {
    {0, 0}, {0, 0}, {0, 0}, {0, 0},
    {0, 0}, {0, 0}, {0, 0}, {0, 0},
    {0, 0}, {0, 0}, {0, 0}, {0, 0},
    {0, 0}, {0, 0}, {0, 0}, {0, 0},
  };
  PARALLEL_FOR(tid, batch_size,
    size_t btid = insert_two_zeros(mask1, mask2, tid);
    TWO_POSITIONS_FOR(
      tmp[8 * p2 + 4 * p1 + 2 * q2 + q1] = ADD(
        tmp[8 * p2 + 4 * p1 + 2 * q2 + q1],
        MUL(
          state[p2 * stride2 + p1 * stride1 + btid],
          CONJ(state[q2 * stride2 + q1 * stride1 + btid])
        )
      );
    )
  )
  TWO_POSITIONS_FOR(
    cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * threadIdx.x] = tmp[8 * p2 + 4 * p1 + 2 * q2 + q1];
  )
  __syncthreads();
  int s = THREADS_NUM / 2;
  while ( s != 0 ) {
    if ( threadIdx.x < s ) {
      TWO_POSITIONS_FOR(
        cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * threadIdx.x] = ADD(
          cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * threadIdx.x],
          cache[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * (threadIdx.x + s)]
        );
      )
    }
    __syncthreads();
    s /= 2;
  }
  if (threadIdx.x == 0) {
    TWO_POSITIONS_FOR(
      density[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * blockIdx.x] = cache[8 * p2 + 4 * p1 + 2 * q2 + q1];
    )
  }
}

extern "C"
int32_t get_q2density(
  const COMPLEX* state,
  COMPLEX* density,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  COMPLEX* device_density;
  COMPLEX* host_density;
  int32_t alloc_status = cudaMalloc(&device_density, 16 * BLOCKS_NUM * sizeof(COMPLEX));
  _get_q2density<<<BLOCKS_NUM, THREADS_NUM>>>(
    state,
    device_density,
    pos2,
    pos1,
    qubits_number
  );
  host_density = (COMPLEX*)malloc(16 * BLOCKS_NUM * sizeof(COMPLEX));
  int32_t memcopy_status = cudaMemcpy(
    host_density,
    device_density,
    16 * BLOCKS_NUM * sizeof(COMPLEX),
    cudaMemcpyDeviceToHost
  );
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    TWO_POSITIONS_FOR(
      density[8 * p2 + 4 * p1 + 2 * q2 + q1] = ADD(
        density[8 * p2 + 4 * p1 + 2 * q2 + q1],
        host_density[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * i]
      );
    )
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
  const COMPLEX* src,
  COMPLEX* dst,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  PARALLEL_FOR(tid, size, dst[tid] = src[tid];)
}

extern "C"
void copy(
  const COMPLEX* src,
  COMPLEX* dst,
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
  const COMPLEX* src,
  COMPLEX* dst,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  PARALLEL_FOR(tid, size,
    dst[tid].x = 2 * src[tid].x;
    dst[tid].y = -2 * src[tid].y;
  )
}

extern "C"
void conj_and_double(
  const COMPLEX* src,
  COMPLEX* dst,
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
  const COMPLEX* src,
  COMPLEX* dst,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  PARALLEL_FOR(tid, size, dst[tid] = ADD(dst[tid], src[tid]);)
}

extern "C"
void add(
  const COMPLEX* src,
  COMPLEX* dst,
  size_t qubits_number
)
{
  _add<<<BLOCKS_NUM, THREADS_NUM>>>(
    src,
    dst,
    qubits_number
  );
}

///////////////////////////////////////////////////////////////////

#ifdef CHECK
  #include <cassert>
  #include <stdio.h>
  int main() {
    int qubits_number = 21;
    COMPLEX* state;
    COMPLEX* host_state;
    host_state = (COMPLEX*)malloc((1 << qubits_number) * sizeof(COMPLEX));
    get_state(&state, qubits_number);
    set2standard(state, qubits_number);
    COMPLEX hadamard[4] = {
      {1 / sqrt(2.f), 0}, {1 / sqrt(2.f), 0},
      {1 / sqrt(2.f), 0}, {-1 / sqrt(2.f), 0}
    };
    COMPLEX cnot[16] = {
      {1, 0}, {0, 0}, {0, 0}, {0, 0},
      {0, 0}, {1, 0}, {0, 0}, {0, 0},
      {0, 0}, {0, 0}, {0, 0}, {1, 0},
      {0, 0}, {0, 0}, {1, 0}, {0, 0}
    };
    COMPLEX diag_cz[4] = { {1, 0}, {1, 0}, {1, 0}, {-1, 0} };
    q1gate(state, hadamard, 0, qubits_number);
    for (int i = 0; i < qubits_number - 2; i++) {
      q2gate(state, cnot, i, i+1, qubits_number);
    }
    // cnot decomposition //
    q1gate(state, hadamard, qubits_number-1, qubits_number);
    q2gate_diag(state, diag_cz, qubits_number-2, qubits_number-1, qubits_number);
    q1gate(state, hadamard, qubits_number-1, qubits_number);
    ////////////////////////
    copy_to_host(state, host_state, qubits_number);
    assert(ABS(SUB(host_state[0], COMPLEXNEW(1 / sqrt(2.f), 0))) < 1e-5);
    assert(ABS(SUB(host_state[(1 << qubits_number)-1], COMPLEXNEW(1 / sqrt(2.f), 0))) < 1e-5);
    for (size_t i = 1; i < (1 << qubits_number) - 1; i++) {
      assert(ABS(SUB(host_state[i], COMPLEXNEW(0, 0))) < 1e-5);
    }
    for (int i = 0; i < qubits_number; i++) {
      COMPLEX q1density[4] = {
        {0, 0}, {0, 0},
        {0, 0}, {0, 0}
      };
      get_q1density(state, q1density, i, qubits_number);
      assert(ABS(SUB(q1density[0], COMPLEXNEW(0.5f, 0))) < 1e-5);
      assert(ABS(SUB(q1density[1], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q1density[2], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q1density[3], COMPLEXNEW(0.5f, 0))) < 1e-5);
    }
    for (int i = 0; i < qubits_number-1; i++) {
      COMPLEX q2density[16] = {
        {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0},
      };
      get_q2density(state, q2density, i, i+1, qubits_number);
      assert(ABS(SUB(q2density[0], COMPLEXNEW(0.5f, 0))) < 1e-5);
      assert(ABS(SUB(q2density[1], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[2], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[3], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[4], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[5], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[6], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[7], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[8], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[9], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[10], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[11], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[12], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[13], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[14], COMPLEXNEW(0, 0))) < 1e-5);
      assert(ABS(SUB(q2density[15], COMPLEXNEW(0.5f, 0))) < 1e-5);
    }
    printf("OK!\n");
    drop_state(state);
    delete[] host_state;
    return 0;
  }
#endif

#ifdef GPU_PROPERTIES
  #include <stdio.h>
  int main() {
    cudaDeviceProp prop;
    int device_number;
    int getdevice_status = cudaGetDevice(&device_number);
    int getdeviceproperties_status = cudaGetDeviceProperties(&prop, device_number);
    if ( getdevice_status != 0) { printf("During 'cudaGetDevice' function call an error with code %d has occurred.", getdevice_status); }
    if ( getdeviceproperties_status != 0) { printf("During 'cudaGetDeviceProperties' function call an error with code %d has occurred.", getdevice_status); }
    printf("Major compute capability: %d\n", prop.major);
    printf("Minor compute capability: %d\n", prop.minor);
    printf("Global memory available in bytes: %d\n", prop.totalGlobalMem);
    return 0;
  }
#endif