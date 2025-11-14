// Copyright (c) OpenMMLab. All rights reserved
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "sigmoid_focal_loss_cuda_kernel.cuh"

void SigmoidFocalLossForwardCUDAKernelLauncher(const at::Tensor input,
                                                const at::Tensor target,
                                                const at::Tensor weight,
                                                at::Tensor output,
                                                const float gamma,
                                                const float alpha) {
  int output_size = output.numel();
  int num_classes = input.size(1);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sigmoid_focal_loss_forward_cuda_kernel", [&] {
        sigmoid_focal_loss_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(),
                weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(), scalar_t(gamma), scalar_t(alpha),
                num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void SigmoidFocalLossBackwardCUDAKernelLauncher(
    const at::Tensor input, const at::Tensor target, const at::Tensor weight,
    at::Tensor grad_input, const float gamma, const float alpha) {
  int output_size = grad_input.numel();
  int num_classes = input.size(1);

  at::cuda::CUDAGuard device_guard(grad_input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sigmoid_focal_loss_backward_cuda_kernel", [&] {
        sigmoid_focal_loss_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(),
                weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
                grad_input.data_ptr<scalar_t>(), scalar_t(gamma),
                scalar_t(alpha), num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}
