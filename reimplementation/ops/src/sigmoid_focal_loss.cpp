// Copyright (c) OpenMMLab. All rights reserved
#include <torch/extension.h>

void SigmoidFocalLossForwardCUDAKernelLauncher(const at::Tensor input,
                                                const at::Tensor target,
                                                const at::Tensor weight,
                                                at::Tensor output,
                                                const float gamma,
                                                const float alpha);

void SigmoidFocalLossBackwardCUDAKernelLauncher(const at::Tensor input,
                                                 const at::Tensor target,
                                                 const at::Tensor weight,
                                                 at::Tensor grad_input,
                                                 const float gamma,
                                                 const float alpha);

void sigmoid_focal_loss_forward_cuda(const at::Tensor input,
                                      const at::Tensor target,
                                      const at::Tensor weight, at::Tensor output,
                                      const float gamma, const float alpha) {
  SigmoidFocalLossForwardCUDAKernelLauncher(input, target, weight, output,
                                             gamma, alpha);
}

void sigmoid_focal_loss_backward_cuda(const at::Tensor input,
                                       const at::Tensor target,
                                       const at::Tensor weight,
                                       at::Tensor grad_input, const float gamma,
                                       const float alpha) {
  SigmoidFocalLossBackwardCUDAKernelLauncher(input, target, weight, grad_input,
                                              gamma, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid_focal_loss_forward", &sigmoid_focal_loss_forward_cuda,
        "sigmoid_focal_loss_forward (CUDA)");
  m.def("sigmoid_focal_loss_backward", &sigmoid_focal_loss_backward_cuda,
        "sigmoid_focal_loss_backward (CUDA)");
}
