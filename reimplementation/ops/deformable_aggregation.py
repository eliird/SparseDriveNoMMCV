"""
Deformable Aggregation CUDA operation.
Pure PyTorch/CUDA implementation without mmcv dependencies.
"""
import torch
from torch.autograd.function import Function, once_differentiable

try:
    from . import deformable_aggregation_ext
    CUDA_EXT_AVAILABLE = True
except ImportError:
    CUDA_EXT_AVAILABLE = False
    deformable_aggregation_ext = None


class DeformableAggregationFunction(Function):
    """Custom CUDA function for efficient deformable feature aggregation.
    
    This function performs bilinear sampling from multi-camera multi-scale feature maps
    at specified 2D locations, then aggregates them using learned attention weights.
    
    The CUDA implementation is significantly faster than PyTorch's grid_sample for this
    specific use case because it fuses the sampling and weighted aggregation operations.
    """
    
    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        """Forward pass of deformable aggregation.
        
        Args:
            mc_ms_feat (Tensor): Flattened multi-camera multi-scale features
                Shape: (bs, num_total_pixels, embed_dims)
            spatial_shape (Tensor): Height and width of each feature map
                Shape: (num_cams, num_levels, 2)
            scale_start_index (Tensor): Starting index for each feature map in flattened tensor
                Shape: (num_cams, num_levels)
            sampling_location (Tensor): 2D sampling locations (normalized to [0, 1])
                Shape: (bs, num_queries, num_pts, num_cams, 2)
            weights (Tensor): Attention weights for aggregation
                Shape: (bs, num_queries, num_pts, num_cams, num_levels, num_groups)
        
        Returns:
            Tensor: Aggregated features of shape (bs, num_queries, embed_dims)
        """
        # Ensure contiguous and correct dtypes
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        
        # Call CUDA kernel
        output = deformable_aggregation_ext.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        
        # Save for backward pass
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """Backward pass of deformable aggregation.
        
        Args:
            grad_output (Tensor): Gradient of loss w.r.t. output
        
        Returns:
            tuple: Gradients w.r.t. (mc_ms_feat, None, None, sampling_location, weights)
        """
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        
        # Ensure contiguous and correct dtypes
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        # Initialize gradients
        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        
        # Call CUDA kernel for backward pass
        deformable_aggregation_ext.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        
        return (
            grad_mc_ms_feat,
            None,  # No gradient for spatial_shape
            None,  # No gradient for scale_start_index
            grad_sampling_location,
            grad_weights,
        )
