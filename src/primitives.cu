#include <cuComplex.h>
#include <stdio.h>
#include "cublas_v2.h"

// TODO: adjust kernel parameters
#define BLOCKS_NUM 128
#define THREADS_NUM 128


// double of single precision
#ifdef F64
# define ADD cuCadd
# define SUB cuCsub
# define MUL cuCmul
# define CONJ cuConj
# define COMPLEX cuDoubleComplex
# define COMPLEXNEW make_cuDoubleComplex
# define ABS cuCabs
# define MATINV cublasZmatinvBatched
#else
# define ADD cuCaddf
# define SUB cuCsubf
# define MUL cuCmulf
# define CONJ cuConjf
# define COMPLEX cuFloatComplex
# define COMPLEXNEW make_cuFloatComplex
# define ABS cuCabsf
# define MATINV cublasCmatinvBatched
#endif

// CUDA errors handler
#define CUDA_CHECK( call )                                                             \
{                                                                                      \
  auto status = (cudaError_t)call;                                                     \
  if ( status != cudaSuccess )                                                         \
  {                                                                                    \
    char *err_str = (char*)malloc(1024 * sizeof(char));                                \
    snprintf(                                                                          \
      err_str,                                                                         \
      1024,                                                                            \
      "CUDA ERROR: call of a function \"%s\" in line %d of file %s failed with %s.\0", \
      #call,                                                                           \
      __LINE__,                                                                        \
      __FILE__,                                                                        \
      cudaGetErrorName(status)                                                         \
    );                                                                                 \
    return err_str;                                                                    \
  }                                                                                    \
}

// cublas errors handler
#define CUBLAS_CHECK( call )                                                                    \
{                                                                                               \
  auto status = (cublasStatus_t)call;                                                           \
  if ( status != CUBLAS_STATUS_SUCCESS )                                                        \
  {                                                                                             \
    char *err_str = (char*)malloc(1024 * sizeof(char));                                         \
    snprintf(                                                                                   \
      err_str,                                                                                  \
      1024,                                                                                     \
      "CUBLAS ERROR: call of a function \"%s\" in line %d of file %s failed with status %d.\0", \
      #call,                                                                                    \
      __LINE__,                                                                                 \
      __FILE__,                                                                                 \
      status                                                                                    \
    );                                                                                          \
    return err_str;                                                                             \
  }                                                                                             \
}

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

// utility macro
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define INSERT_ZERO(mask, offset)                         \
(((mask) & (offset)) << 1) | ((~(mask)) & (offset))       \


// const memory for gates
__constant__ COMPLEX q1g[4];
__constant__ COMPLEX q2g[16];
__constant__ COMPLEX q2dg[4];

// matrix inverse
static char* inv(const COMPLEX* A, COMPLEX* Ainv, size_t size) {
  cublasHandle_t handle;
  int* info;
  int host_info;
  CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
  COMPLEX** Abatch;
  COMPLEX** Ainvbatch;
  CUDA_CHECK(cudaMalloc(&Abatch, sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&Ainvbatch, sizeof(void*)));
  CUDA_CHECK(cudaMemcpy(Abatch, &A, sizeof(void*), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Ainvbatch, &Ainv, sizeof(void*), cudaMemcpyHostToDevice));
  CUBLAS_CHECK(cublasCreate_v2(&handle));
  CUBLAS_CHECK(MATINV(handle, size, Abatch, size, Ainvbatch, size, info, 1));
  CUDA_CHECK(cudaMemcpy(&host_info, info, sizeof(int), cudaMemcpyDeviceToHost));
  if ( host_info != 0 ) {
    char* err_msg = (char*)malloc(1024 * sizeof(char));
    snprintf(err_msg, 1024, "U(%d, %d) is zero.", host_info, host_info);
    return err_msg;
  }
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(info));
  CUDA_CHECK(cudaFree(Ainvbatch));
  CUDA_CHECK(cudaFree(Abatch));
  return 0;
}

// allocate an uninitialized state on the device
extern "C"
const char* get_state (
  COMPLEX** state,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  CUDA_CHECK(cudaMalloc(state, size * sizeof(COMPLEX)));
  return 0;
}

// copy a state to the host
extern "C"
const char* copy_to_host (
  COMPLEX* state,
  COMPLEX* host_state,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  CUDA_CHECK(cudaMemcpy(host_state, state, size * sizeof(COMPLEX), cudaMemcpyDeviceToHost));
  return 0;
}

// drop state on the device
extern "C"
const char* drop_state(
  COMPLEX* state
)
{
  CUDA_CHECK(cudaFree(state));
  return 0;
}

// initialize state to the standard one
static __global__ void _set2standard (
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
static __global__ void _q1grad (
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
    size_t btid = INSERT_ZERO(mask, tid);
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
const char* q1grad (
  const COMPLEX* fwd,
  const COMPLEX* bwd,
  COMPLEX* grad,
  size_t pos,
  size_t qubits_number
)
{
  COMPLEX* device_grad;
  COMPLEX* host_grad;
  CUDA_CHECK(cudaMalloc(&device_grad, 4 * BLOCKS_NUM * sizeof(COMPLEX)));
  _q1grad<<<BLOCKS_NUM, THREADS_NUM>>>(
    fwd,
    bwd,
    device_grad,
    pos,
    qubits_number
  );
  host_grad = (COMPLEX*)malloc(4 * BLOCKS_NUM * sizeof(COMPLEX));
  CUDA_CHECK(cudaMemcpy(
    host_grad,
    device_grad,
    4 * BLOCKS_NUM * sizeof(COMPLEX),
    cudaMemcpyDeviceToHost
  ));
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    ONE_POSITION_FOR(
      grad[2 * p + q] = ADD(
        grad[2 * p + q],
        host_grad[2 * p + q + 4 * i]
      );
    )
  }
  free(host_grad);
  CUDA_CHECK(cudaFree(device_grad));
  return 0;
}

// computes q2 gate gradient from fwd and bwd "states"
static __global__ void _q2grad (
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
  size_t max_mask = MIN(mask1, mask2);
  size_t min_mask = MAX(mask1, mask2);
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
    size_t btid = INSERT_ZERO(min_mask, tid);
    btid = INSERT_ZERO(max_mask, btid);
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
const char* q2grad (
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
  CUDA_CHECK(cudaMalloc(&device_grad, 16 * BLOCKS_NUM * sizeof(COMPLEX)));
  _q2grad<<<BLOCKS_NUM, THREADS_NUM>>>(
    fwd,
    bwd,
    device_grad,
    pos2,
    pos1,
    qubits_number
  );
  host_grad = (COMPLEX*)malloc(16 * BLOCKS_NUM * sizeof(COMPLEX));
  CUDA_CHECK(cudaMemcpy(
    host_grad,
    device_grad,
    16 * BLOCKS_NUM * sizeof(COMPLEX),
    cudaMemcpyDeviceToHost
  ));
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    TWO_POSITIONS_FOR(
      grad[8 * p2 + 4 * p1 + 2 * q2 + q1] = ADD(
        grad[8 * p2 + 4 * p1 + 2 * q2 + q1],
        host_grad[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * i]
      );
    )
  }
  free(host_grad);
  CUDA_CHECK(cudaFree(device_grad));
  return 0;
}

// computes q2 diagonal gate gradient from fwd and bwd "states"
static __global__ void _q2grad_diag (
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
  size_t max_mask = MIN(mask1, mask2);
  size_t min_mask = MAX(mask1, mask2);
  size_t stride1 = 1 << pos1;
  size_t stride2 = 1 << pos2;
  size_t batch_size = size >> 2;
  COMPLEX tmp[4] = { {0, 0}, {0, 0}, {0, 0}, {0, 0} };
  PARALLEL_FOR(tid, batch_size,
    size_t btid = INSERT_ZERO(min_mask, tid);
    btid = INSERT_ZERO(max_mask, btid);
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
const char* q2grad_diag (
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
  CUDA_CHECK(cudaMalloc(&device_grad, 4 * BLOCKS_NUM * sizeof(COMPLEX)));
  _q2grad_diag<<<BLOCKS_NUM, THREADS_NUM>>>(
    fwd,
    bwd,
    device_grad,
    pos2,
    pos1,
    qubits_number
  );
  host_grad = (COMPLEX*)malloc(4 * BLOCKS_NUM * sizeof(COMPLEX));
  CUDA_CHECK(cudaMemcpy(
    host_grad,
    device_grad,
    4 * BLOCKS_NUM * sizeof(COMPLEX),
    cudaMemcpyDeviceToHost
  ));
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    ONE_POSITION_FOR(
      grad[2 * p + q] = ADD(
        grad[2 * p + q],
        host_grad[2 * p + q + 4 * i]
      );
    )
  }
  free(host_grad);
  CUDA_CHECK(cudaFree(device_grad));
  return 0;
}

// sets state from host
extern "C"
const char* set_from_host (
  COMPLEX* device_state,
  const COMPLEX* host_state,
  size_t qubits_number
)
{
  CUDA_CHECK(cudaMemcpy(
    device_state,
    host_state,
    (1 << qubits_number) * sizeof(COMPLEX),
    cudaMemcpyHostToDevice
  ));
  return 0;
}

// one qubits gate application
static __global__ void _q1gate(
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
    size_t btid = INSERT_ZERO(mask, tid);
    COMPLEX tmp[2] = { {0, 0}, {0, 0} };
    ONE_POSITION_FOR(
      tmp[p] = ADD(tmp[p], MUL(q1g[2 * p + q], state[stride * q + btid]));
    )
    state[btid] = tmp[0];
    state[btid + stride] = tmp[1];
  )
}

extern "C"
const char* q1gate(
  COMPLEX* state,
  const COMPLEX* gate,
  size_t pos,
  size_t qubits_number
)
{
  CUDA_CHECK(cudaMemcpyToSymbol(q1g, gate, 4 * sizeof(COMPLEX)));
  _q1gate<<<BLOCKS_NUM, THREADS_NUM>>>(state, pos, qubits_number);
  return 0;
}

extern "C"
const char* q1gate_inv(
  COMPLEX* state,
  const COMPLEX* gate,
  size_t pos,
  size_t qubits_number
)
{
  // TODO: reduce number of CUDA mallocs
  COMPLEX* d_inv_gate;
  COMPLEX* d_gate;
  cudaMalloc(&d_gate, 4 * sizeof(COMPLEX));
  cudaMalloc(&d_inv_gate, 4 * sizeof(COMPLEX));
  cudaMemcpy(d_gate, gate, 4 * sizeof(COMPLEX), cudaMemcpyHostToDevice);
  char* status = inv(d_gate, d_inv_gate, 2);
  if (status != 0) {
    return status; 
  }
  CUDA_CHECK(cudaMemcpyToSymbol(q1g, d_inv_gate, 4 * sizeof(COMPLEX)));
  CUDA_CHECK(cudaFree(d_inv_gate));
  CUDA_CHECK(cudaFree(d_gate));
  _q1gate<<<BLOCKS_NUM, THREADS_NUM>>>(state, pos, qubits_number);
  return 0;
}

// two qubits gate application
static __global__ void _q2gate(
  COMPLEX* state,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  size_t mask1 =  SIZE_MAX << pos1;
  size_t mask2 =  SIZE_MAX << pos2;
  size_t max_mask = MIN(mask1, mask2);
  size_t min_mask = MAX(mask1, mask2);
  size_t stride1 = 1 << pos1;
  size_t stride2 = 1 << pos2;
  size_t batch_size = size >> 2;
  PARALLEL_FOR(tid, batch_size,
    size_t btid = INSERT_ZERO(min_mask, tid);
    btid = INSERT_ZERO(max_mask, btid);
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
const char* q2gate(
  COMPLEX* state,
  const COMPLEX* gate,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  CUDA_CHECK(cudaMemcpyToSymbol(q2g, gate, 16 * sizeof(COMPLEX)));
  _q2gate<<<BLOCKS_NUM, THREADS_NUM>>>(state, pos2, pos1, qubits_number);
  return 0;
}

extern "C"
const char* q2gate_inv(
  COMPLEX* state,
  const COMPLEX* gate,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  // TODO: reduce number of CUDA mallocs
  COMPLEX* d_inv_gate;
  COMPLEX* d_gate;
  cudaMalloc(&d_gate, 16 * sizeof(COMPLEX));
  cudaMalloc(&d_inv_gate, 16 * sizeof(COMPLEX));
  cudaMemcpy(d_gate, gate, 16 * sizeof(COMPLEX), cudaMemcpyHostToDevice);
  char* status = inv(d_gate, d_inv_gate, 4);
  if (status != 0) {
    return status; 
  }
  CUDA_CHECK(cudaMemcpyToSymbol(q2g, d_inv_gate, 16 * sizeof(COMPLEX)));
  CUDA_CHECK(cudaFree(d_inv_gate));
  CUDA_CHECK(cudaFree(d_gate));
  _q2gate<<<BLOCKS_NUM, THREADS_NUM>>>(state, pos2, pos1, qubits_number);
  return 0;
}

// two qubits diagonal gate application
static __global__ void _q2gate_diag(
  COMPLEX* state,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  size_t size = 1 << qubits_number;
  size_t mask1 =  SIZE_MAX << pos1;
  size_t mask2 =  SIZE_MAX << pos2;
  size_t max_mask = MIN(mask1, mask2);
  size_t min_mask = MAX(mask1, mask2);
  size_t stride1 = 1 << pos1;
  size_t stride2 = 1 << pos2;
  size_t batch_size = size >> 2;
  PARALLEL_FOR(tid, batch_size,
    size_t btid = INSERT_ZERO(min_mask, tid);
    btid = INSERT_ZERO(max_mask, btid);
    state[btid] = MUL(q2dg[0], state[btid]);
    state[btid + stride1] = MUL(q2dg[1], state[btid + stride1]);
    state[btid + stride2] = MUL(q2dg[2], state[btid + stride2]);
    state[btid + stride1 + stride2] = MUL(q2dg[3], state[btid + stride1 + stride2]);
  )
}

extern "C"
const char* q2gate_diag(
  COMPLEX* state,
  const COMPLEX* gate,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  CUDA_CHECK(cudaMemcpyToSymbol(q2dg, gate, 4 * sizeof(COMPLEX)));
  _q2gate_diag<<<BLOCKS_NUM, THREADS_NUM>>>(state, pos2, pos1, qubits_number);
  return 0;
}

// one qubit density matrix computation
static __global__ void _get_q1density(
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
    size_t btid = INSERT_ZERO(mask, tid);
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
const char* get_q1density(
  const COMPLEX* state,
  COMPLEX* density,
  size_t pos,
  size_t qubits_number
)
{
  COMPLEX* device_density;
  COMPLEX* host_density;
  CUDA_CHECK(cudaMalloc(&device_density, 4 * BLOCKS_NUM * sizeof(COMPLEX)));
  _get_q1density<<<BLOCKS_NUM, THREADS_NUM>>>(
    state,
    device_density,
    pos,
    qubits_number
  );
  host_density = (COMPLEX*)malloc(4 * BLOCKS_NUM * sizeof(COMPLEX));
  CUDA_CHECK(cudaMemcpy(
    host_density,
    device_density,
    4 * BLOCKS_NUM * sizeof(COMPLEX),
    cudaMemcpyDeviceToHost
  ));
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    ONE_POSITION_FOR(
      density[2 * p + q] = ADD(
        density[2 * p + q],
        host_density[2 * p + q + 4 * i]
      );
    )
  }
  free(host_density);
  CUDA_CHECK(cudaFree(device_density));
  return 0;
}

// two qubit density matrix computation
static __global__ void _get_q2density(
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
  size_t max_mask = MIN(mask1, mask2);
  size_t min_mask = MAX(mask1, mask2);
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
    size_t btid = INSERT_ZERO(min_mask, tid);
    btid = INSERT_ZERO(max_mask, btid);
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
const char* get_q2density(
  const COMPLEX* state,
  COMPLEX* density,
  size_t pos2,
  size_t pos1,
  size_t qubits_number
)
{
  COMPLEX* device_density;
  COMPLEX* host_density;
  CUDA_CHECK(cudaMalloc(&device_density, 16 * BLOCKS_NUM * sizeof(COMPLEX)));
  _get_q2density<<<BLOCKS_NUM, THREADS_NUM>>>(
    state,
    device_density,
    pos2,
    pos1,
    qubits_number
  );
  host_density = (COMPLEX*)malloc(16 * BLOCKS_NUM * sizeof(COMPLEX));
  CUDA_CHECK(cudaMemcpy(
    host_density,
    device_density,
    16 * BLOCKS_NUM * sizeof(COMPLEX),
    cudaMemcpyDeviceToHost
  ));
  for (int i = 0; i < BLOCKS_NUM; i ++) {
    TWO_POSITIONS_FOR(
      density[8 * p2 + 4 * p1 + 2 * q2 + q1] = ADD(
        density[8 * p2 + 4 * p1 + 2 * q2 + q1],
        host_density[8 * p2 + 4 * p1 + 2 * q2 + q1 + 16 * i]
      );
    )
  }
  free(host_density);
  CUDA_CHECK(cudaFree(device_density));
  return 0;
}

// copy of a state
static __global__ void _copy(
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
static __global__ void _conj_and_double(
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

static __global__ void _add(
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
  // This checks correctness of the ghz state preparation
  void ghz_test() {
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
    free(host_state);
  }
  // this tests matrix inversion
  void inv_test() {
    COMPLEX h_A[9] = {
      {1., 1.1},  {2., 2.},  {3., 3.},
      {1.2, 2.3}, {3.2, 0.}, {1., 1.5},
      {0., 2.1},  {2., 4.},  {2.11, 3.44},
    };
    COMPLEX h_Ainv[9];
    COMPLEX* d_A;
    COMPLEX* d_Ainv;
    COMPLEX M[9] ={
      {0., 0.}, {0., 0.}, {0., 0.},
      {0., 0.}, {0., 0.}, {0., 0.},
      {0., 0.}, {0., 0.}, {0., 0.}
    };
    cudaMalloc(&d_A, 9 * sizeof(COMPLEX));
    cudaMalloc(&d_Ainv, 9 * sizeof(COMPLEX));
    cudaMemcpy(d_A, h_A, 9 * sizeof(COMPLEX), cudaMemcpyHostToDevice);
    inv(d_A, d_Ainv, 3);
    cudaMemcpy(h_Ainv, d_Ainv, 9 * sizeof(COMPLEX), cudaMemcpyDeviceToHost);
    for (int q = 0; q < 3; q++) {
      for (int p = 0; p < 3; p++) {
        for (int m = 0; m < 3; m++) {
          M[3 * q + p] = ADD(M[3 * q + p], MUL(h_A[3 * q + m], h_Ainv[3 * m + p]));
        }
      }
    }
    assert(ABS(SUB(M[0], COMPLEXNEW(1, 0))) < 1e-5);
    assert(ABS(SUB(M[1], COMPLEXNEW(0, 0))) < 1e-5);
    assert(ABS(SUB(M[2], COMPLEXNEW(0, 0))) < 1e-5);
    assert(ABS(SUB(M[3], COMPLEXNEW(0, 0))) < 1e-5);
    assert(ABS(SUB(M[4], COMPLEXNEW(1, 0))) < 1e-5);
    assert(ABS(SUB(M[5], COMPLEXNEW(0, 0))) < 1e-5);
    assert(ABS(SUB(M[6], COMPLEXNEW(0, 0))) < 1e-5);
    assert(ABS(SUB(M[7], COMPLEXNEW(0, 0))) < 1e-5);
    assert(ABS(SUB(M[8], COMPLEXNEW(1, 0))) < 1e-5);
    cudaFree(d_A);
    cudaFree(d_Ainv);
    cudaFree(M);
  }
  int main() {
    ghz_test();
    inv_test();
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