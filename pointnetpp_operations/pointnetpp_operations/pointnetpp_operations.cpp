
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>


#define CHECK_CUDA(x)                                          \
    do {                                                         \
    AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor"); \
    } while (0)

#define CHECK_CONTIGUOUS(x)                                         \
    do {                                                              \
    AT_CHECK(x.is_contiguous(), #x " must be a contiguous tensor"); \
    } while (0)

#define CHECK_IS_INT(x)                              \
    do {                                               \
    AT_CHECK(x.scalar_type() == at::ScalarType::Int, \
             #x " must be an int tensor");           \
    } while (0)

#define CHECK_IS_FLOAT(x)                              \
    do {                                                 \
    AT_CHECK(x.scalar_type() == at::ScalarType::Float, \
             #x " must be a float tensor");            \
    } while (0)


// #######################
// Furthest Point Sampling

void FurthestPointSamplingKernelLauncher(const int b, const int n, const int m, const float *dataset, float *temp, int *idxs);

at::Tensor furthest_point_sampling_cuda(const at::Tensor points, const int nsamples) {
    CHECK_CONTIGUOUS(points);
    CHECK_IS_FLOAT(points);

    at::Tensor output = torch::zeros({points.size(0), nsamples}, at::device(points.device()).dtype(at::ScalarType::Int));
    at::Tensor tmp = torch::full({points.size(0), points.size(1)}, 1e10, at::device(points.device()).dtype(at::ScalarType::Float));

    FurthestPointSamplingKernelLauncher(points.size(0), points.size(1), nsamples, points.data<float>(),
                                        tmp.data<float>(), output.data<int>());

    return output;
}


// ############
// Group Points

void GroupPointsKernelLauncher(const int b, const int c, const int n, const int npoints, const int nsample,
                               const float *points, const int *idx, float *out);

void GroupPointsGradKernelLauncher(const int b, const int c, const int n, const int npoints,
                                   const int nsample, const float *grad_out, const int *idx, float *grad_points);

at::Tensor group_points_cuda(at::Tensor points, at::Tensor idx) {
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);

    at::Tensor output = torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                                      at::device(points.device()).dtype(at::ScalarType::Float));

    GroupPointsKernelLauncher(points.size(0), points.size(1), points.size(2),
                              idx.size(1), idx.size(2), points.data<float>(),
                              idx.data<int>(), output.data<float>());

    return output;
}

at::Tensor group_points_grad_cuda(at::Tensor grad_out, at::Tensor idx, const int n) {
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);

    at::Tensor output = torch::zeros({grad_out.size(0), grad_out.size(1), n}, at::device(grad_out.device()).dtype(at::ScalarType::Float));

    GroupPointsGradKernelLauncher(grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
                                  grad_out.data<float>(), idx.data<int>(), output.data<float>());

    return output;
}


// #################
// Three Interpolate

void ThreeNNKernelLauncher(const int b, const int n, const int m, const float *unknown,
                            const float *known, float *dist2, int *idx);
void ThreeInterpolateKernelLauncher(const int b, const int c, const int m, int n, const float *points,
                                    const int *idx, const float *weight, float *out);
void ThreeInterpolateGradKernelLauncher(const int b, const int c, const int n, const int m, const float *grad_out,
                                        const int *idx, const float *weight, float *grad_points);

std::vector<at::Tensor> three_nn_cuda(const at::Tensor unknowns, const at::Tensor knows) {
    CHECK_CONTIGUOUS(unknowns);
    CHECK_CONTIGUOUS(knows);
    CHECK_IS_FLOAT(unknowns);
    CHECK_IS_FLOAT(knows);

    at::Tensor idx = torch::zeros({unknowns.size(0), unknowns.size(1), 3}, at::device(unknowns.device()).dtype(at::ScalarType::Int));
    at::Tensor dist2 = torch::zeros({unknowns.size(0), unknowns.size(1), 3}, at::device(unknowns.device()).dtype(at::ScalarType::Float));

    ThreeNNKernelLauncher(unknowns.size(0), unknowns.size(1), knows.size(1),
                          unknowns.data<float>(), knows.data<float>(),
                          dist2.data<float>(), idx.data<int>());

    return {dist2, idx};
}

at::Tensor three_interpolate_cuda(const at::Tensor points, const at::Tensor idx, const at::Tensor weight) {
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(weight);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);
    CHECK_IS_FLOAT(weight);

    at::Tensor output = torch::zeros({points.size(0), points.size(1), idx.size(1)}, at::device(points.device()).dtype(at::ScalarType::Float));

    ThreeInterpolateKernelLauncher(
        points.size(0), points.size(1), points.size(2), idx.size(1),
        points.data<float>(), idx.data<int>(), weight.data<float>(),
        output.data<float>());

    return output;
}

at::Tensor three_interpolate_grad_cuda(const at::Tensor grad_out, const at::Tensor idx, const at::Tensor weight, const int m) {
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(weight);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);
    CHECK_IS_FLOAT(weight);

    at::Tensor output = torch::zeros({grad_out.size(0), grad_out.size(1), m}, at::device(grad_out.device()).dtype(at::ScalarType::Float));

    ThreeInterpolateGradKernelLauncher(
        grad_out.size(0), grad_out.size(1), grad_out.size(2), m,
        grad_out.data<float>(), idx.data<int>(), weight.data<float>(),
        output.data<float>());

    return output;
}


// ############
// Gather Points

void GatherPointsKernelLauncher(const int b, const int c, const int n, const int npoints,
                                const float *points, const int *idx, float *out);
void GatherPointsGradKernelLauncher(const int b, const int c, const int n, const int npoints,
                                    const float *grad_out, const int *idx, float *grad_points);
at::Tensor gather_points_cuda(const at::Tensor points, const at::Tensor idx) {
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);

    at::Tensor output = torch::zeros({points.size(0), points.size(1), idx.size(1)}, at::device(points.device()).dtype(at::ScalarType::Float));

    GatherPointsKernelLauncher(points.size(0), points.size(1), points.size(2), idx.size(1), points.data<float>(),
                               idx.data<int>(), output.data<float>());

    return output;
}

at::Tensor gather_points_grad_cuda(const at::Tensor grad_out, const at::Tensor idx, const int n) {
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);

    at::Tensor output = torch::zeros({grad_out.size(0), grad_out.size(1), n}, at::device(grad_out.device()).dtype(at::ScalarType::Float));


    GatherPointsGradKernelLauncher(grad_out.size(0), grad_out.size(1), n, idx.size(1), grad_out.data<float>(),
                                   idx.data<int>(), output.data<float>());

    return output;
}


// ##########
// Query Ball

void QueryBallPointKernelLauncher(const int b, const int n, const int m, const float radius, const int nsample,
                                  const float *new_xyz, const float *xyz, int *idx);

at::Tensor ball_query_cuda(const at::Tensor new_xyz, const at::Tensor xyz, const float radius, const int nsample) {
    CHECK_CONTIGUOUS(new_xyz);
    CHECK_CONTIGUOUS(xyz);
    CHECK_IS_FLOAT(new_xyz);
    CHECK_IS_FLOAT(xyz);

    auto idx = torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                             at::device(new_xyz.device()).dtype(at::ScalarType::Int));

    QueryBallPointKernelLauncher(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                 radius, nsample,
                                 new_xyz.data<float>(), xyz.data<float>(), idx.data<int>());

    return idx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("furthest_point_sampling_cuda", &furthest_point_sampling_cuda, "furthest_point_sampling_cuda");
    m.def("group_points_cuda", &group_points_cuda, "group_points_cuda");
    m.def("group_points_grad_cuda", &group_points_grad_cuda, "group_points_grad_cuda");
    m.def("three_nn_cuda", &three_nn_cuda, "three_nn_cuda");
    m.def("three_interpolate_cuda", &three_interpolate_cuda, "three_interpolate_cuda");
    m.def("three_interpolate_grad_cuda", &three_interpolate_grad_cuda, "three_interpolate_grad_cuda");
    m.def("gather_points_cuda", &gather_points_cuda, "gather_points_cuda");
    m.def("gather_points_grad_cuda", &gather_points_grad_cuda, "gather_points_grad_cuda");
    m.def("ball_query_cuda", &ball_query_cuda, "ball_query_cuda");
}
