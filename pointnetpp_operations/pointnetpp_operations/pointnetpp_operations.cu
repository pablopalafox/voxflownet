#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_CHECK_ERRORS()                                           \
  do {                                                                \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
              __FILE__);                                              \
      exit(-1);                                                       \
    }                                                                 \
  } while (0)


#define TOTAL_THREADS 512

inline int opt_n_threads(const int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

inline dim3 opt_block_config(const int x, const int y) {
    const int x_threads = opt_n_threads(x);
    const int y_threads = max(min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
    dim3 block_config(x_threads, y_threads, 1);

    return block_config;
}



// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
__global__ void gather_points_kernel(int b, int c, int n, int m,
                                     const float *__restrict__ points,
                                     const int *__restrict__ idx,
                                     float *__restrict__ out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}

void GatherPointsKernelLauncher(const int b, const int c, const int n, const int npoints,
                                const float *points, const int *idx, float *out) {
  gather_points_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                         at::cuda::getCurrentCUDAStream()>>>(b, c, n, npoints, points, idx, out);
  CUDA_CHECK_ERRORS();
}
// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
__global__ void gather_points_grad_kernel(int b, int c, int n, int m,
                                          const float *__restrict__ grad_out,
                                          const int *__restrict__ idx,
                                          float *__restrict__ grad_points) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        atomicAdd(grad_points + (i * c + l) * n + a,
                  grad_out[(i * c + l) * m + j]);
      }
    }
  }
}
void GatherPointsGradKernelLauncher(const int b, const int c, const int n, const int npoints,
                                    const float *grad_out, const int *idx, float *grad_points) {
  gather_points_grad_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      b, c, n, npoints, grad_out, idx, grad_points);
  CUDA_CHECK_ERRORS();
}
__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}
// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    int b, int n, int m, const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs) {
  if (m <= 0) return;
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];
  int batch_index = blockIdx.x;
  dataset += batch_index * n * 3;
  temp += batch_index * n;
  idxs += batch_index * m;
  int tid = threadIdx.x;
  const int stride = block_size;
  int old = 0;
  if (threadIdx.x == 0) idxs[0] = old;
  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * 3 + 0];
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      float x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];
      float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      if (mag <= 1e-3) continue;
      float d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      float d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();
    if (block_size >= 512) {
      if (tid < 256) {
        __update(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }
    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
  }
}
void FurthestPointSamplingKernelLauncher(const int b, const int n, const int m, const float *dataset, float *temp, int *idxs) {
  unsigned int n_threads = opt_n_threads(n);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (n_threads) {
    case 512:
      furthest_point_sampling_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 256:
      furthest_point_sampling_kernel<256>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 128:
      furthest_point_sampling_kernel<128>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 64:
      furthest_point_sampling_kernel<64>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 32:
      furthest_point_sampling_kernel<32>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 16:
      furthest_point_sampling_kernel<16>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 8:
      furthest_point_sampling_kernel<8>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 4:
      furthest_point_sampling_kernel<4>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 2:
      furthest_point_sampling_kernel<2>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 1:
      furthest_point_sampling_kernel<1>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    default:
      furthest_point_sampling_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
  }
  CUDA_CHECK_ERRORS();
}



// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
__global__ void group_points_kernel(int b, int c, int n, int npoints,
                                    int nsample,
                                    const float *__restrict__ points,
                                    const int *__restrict__ idx,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  points += batch_index * n * c;
  idx += batch_index * npoints * nsample;
  out += batch_index * npoints * nsample * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      out[(l * npoints + j) * nsample + k] = points[l * n + ii];
    }
  }
}

void GroupPointsKernelLauncher(const int b, const int c, const int n, const int npoints, const int nsample,
                               const float *points, const int *idx, float *out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  group_points_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
      b, c, n, npoints, nsample, points, idx, out);

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)
__global__ void group_points_grad_kernel(int b, int c, int n, int npoints,
                                         int nsample,
                                         const float *__restrict__ grad_out,
                                         const int *__restrict__ idx,
                                         float *__restrict__ grad_points) {
  int batch_index = blockIdx.x;
  grad_out += batch_index * npoints * nsample * c;
  idx += batch_index * npoints * nsample;
  grad_points += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      atomicAdd(grad_points + l * n + ii,
                grad_out[(l * npoints + j) * nsample + k]);
    }
  }
}

void GroupPointsGradKernelLauncher(const int b, const int c, const int n, const int npoints,
                                   const int nsample, const float *grad_out, const int *idx, float *grad_points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  group_points_grad_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
      b, c, n, npoints, nsample, grad_out, idx, grad_points);

  CUDA_CHECK_ERRORS();
}



// input: unknown(b, n, 3) known(b, m, 3)
// output: dist2(b, n, 3), idx(b, n, 3)
__global__ void three_nn_kernel(int b, int n, int m,
                                const float *__restrict__ unknown,
                                const float *__restrict__ known,
                                float *__restrict__ dist2,
                                int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  unknown += batch_index * n * 3;
  known += batch_index * m * 3;
  dist2 += batch_index * n * 3;
  idx += batch_index * n * 3;

  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int j = index; j < n; j += stride) {
    float ux = unknown[j * 3 + 0];
    float uy = unknown[j * 3 + 1];
    float uz = unknown[j * 3 + 2];

    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < m; ++k) {
      float x = known[k * 3 + 0];
      float y = known[k * 3 + 1];
      float z = known[k * 3 + 2];
      float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
      if (d < best1) {
        best3 = best2;
        besti3 = besti2;
        best2 = best1;
        besti2 = besti1;
        best1 = d;
        besti1 = k;
      } else if (d < best2) {
        best3 = best2;
        besti3 = besti2;
        best2 = d;
        besti2 = k;
      } else if (d < best3) {
        best3 = d;
        besti3 = k;
      }
    }
    dist2[j * 3 + 0] = best1;
    dist2[j * 3 + 1] = best2;
    dist2[j * 3 + 2] = best3;

    idx[j * 3 + 0] = besti1;
    idx[j * 3 + 1] = besti2;
    idx[j * 3 + 2] = besti3;
  }
}

void ThreeNNKernelLauncher(int b, int n, int m, const float *unknown,
                             const float *known, float *dist2, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  three_nn_kernel<<<b, opt_n_threads(n), 0, stream>>>(b, n, m, unknown, known,
                                                      dist2, idx);

  CUDA_CHECK_ERRORS();
}

// input: points(b, c, m), idx(b, n, 3), weight(b, n, 3)
// output: out(b, c, n)
__global__ void three_interpolate_kernel(int b, int c, int m, int n,
                                         const float *__restrict__ points,
                                         const int *__restrict__ idx,
                                         const float *__restrict__ weight,
                                         float *__restrict__ out) {
  int batch_index = blockIdx.x;
  points += batch_index * m * c;

  idx += batch_index * n * 3;
  weight += batch_index * n * 3;

  out += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * n; i += stride) {
    const int l = i / n;
    const int j = i % n;
    float w1 = weight[j * 3 + 0];
    float w2 = weight[j * 3 + 1];
    float w3 = weight[j * 3 + 2];

    int i1 = idx[j * 3 + 0];
    int i2 = idx[j * 3 + 1];
    int i3 = idx[j * 3 + 2];

    out[i] = points[l * m + i1] * w1 + points[l * m + i2] * w2 +
             points[l * m + i3] * w3;
  }
}

void ThreeInterpolateKernelLauncher(int b, int c, int m, int n,
                                      const float *points, const int *idx,
                                      const float *weight, float *out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  three_interpolate_kernel<<<b, opt_block_config(n, c), 0, stream>>>(
      b, c, m, n, points, idx, weight, out);

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, n), idx(b, n, 3), weight(b, n, 3)
// output: grad_points(b, c, m)

__global__ void three_interpolate_grad_kernel(
    int b, int c, int n, int m, const float *__restrict__ grad_out,
    const int *__restrict__ idx, const float *__restrict__ weight,
    float *__restrict__ grad_points) {
  int batch_index = blockIdx.x;
  grad_out += batch_index * n * c;
  idx += batch_index * n * 3;
  weight += batch_index * n * 3;
  grad_points += batch_index * m * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * n; i += stride) {
    const int l = i / n;
    const int j = i % n;
    float w1 = weight[j * 3 + 0];
    float w2 = weight[j * 3 + 1];
    float w3 = weight[j * 3 + 2];

    int i1 = idx[j * 3 + 0];
    int i2 = idx[j * 3 + 1];
    int i3 = idx[j * 3 + 2];

    atomicAdd(grad_points + l * m + i1, grad_out[i] * w1);
    atomicAdd(grad_points + l * m + i2, grad_out[i] * w2);
    atomicAdd(grad_points + l * m + i3, grad_out[i] * w3);
  }
}

void ThreeInterpolateGradKernelLauncher(int b, int c, int n, int m,
                                           const float *grad_out,
                                           const int *idx, const float *weight,
                                           float *grad_points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  three_interpolate_grad_kernel<<<b, opt_block_config(n, c), 0, stream>>>(
      b, c, n, m, grad_out, idx, weight, grad_points);

  CUDA_CHECK_ERRORS();
}



// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel(const int b, const int n, const int m, const float radius, const int nsample,
                                             const float *__restrict__ new_xyz, const float *__restrict__ xyz,
                                             int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

void QueryBallPointKernelLauncher(const int b, const int n, const int m, const float radius, const int nsample, const float *new_xyz, const float *xyz, int *idx) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);
    CUDA_CHECK_ERRORS();
}
