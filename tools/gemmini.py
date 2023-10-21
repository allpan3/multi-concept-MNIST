
import torch
import torch.nn as nn

class ConvParams:
    def __init__(self, kernel_size: int, batch_size: int,
                 in_dim: int, in_channels: int,
                 out_channels: int,
                 stride: int, padding: int,
                 bias: int,
                 depthwise: bool,
                 pool_size: int,
                 pool_stride: int,
                 pool_padding: int,
                 output_scale: int):
                 # gemmini_dim: int):

        assert not depthwise or in_channels == out_channels

        self.batch_size = batch_size
        self.in_row_dim = in_dim
        self.in_col_dim = in_dim
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.depthwise = depthwise
        self.output_scale = output_scale
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

        self.out_row_dim = (in_dim - kernel_size + 2*padding) // stride + 1
        self.out_col_dim = (in_dim - kernel_size + 2*padding) // stride + 1

        self.out_dim_pooled = (self.out_row_dim - pool_size + 2*pool_padding) // pool_stride + 1

        self.n_patches = self.out_row_dim * self.out_col_dim * batch_size

        if depthwise:
            self.patch_size = kernel_size * kernel_size
        else:
            self.patch_size = in_channels * kernel_size * kernel_size

        self.is_pooled = pool_size > 1 or pool_stride > 1

class FcParams:
    def __init__(self, batch_size: int, in_features: int, out_features: int, bias: bool, output_scale: int): # , gemmini_dim: int):
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.output_scale = output_scale
        
        # self.gemmini_dim = gemmini_dim

def get_params(layer: nn.Module, batch_size=None, in_dim=None, pool_stride=None, pool_size=None, pool_padding=None):
    if isinstance(layer, nn.Conv2d):
        if in_dim is None or pool_stride is None or pool_size is None:
            raise Exception("Need in_dim")
        return ConvParams(kernel_size=layer.kernel_size[0], batch_size=batch_size,
                             in_dim=in_dim, in_channels=layer.in_channels,
                             out_channels=layer.out_channels, depthwise=layer.groups > 1,
                             stride=layer.stride[0], padding=layer.padding[0], bias=layer.bias is not None,
                             pool_size=pool_size, pool_stride=pool_stride, pool_padding=pool_padding,
                             output_scale = layer.output_scale if hasattr(layer, 'output_scale') else 1)
    elif isinstance(layer, nn.Linear):
        return FcParams(batch_size=batch_size, in_features=layer.in_features, out_features=layer.out_features, bias=layer.bias is not None, output_scale=layer.output_scale if hasattr(layer, 'output_scale') else 1)

def filter2col(x: torch.Tensor, params: ConvParams) -> torch.Tensor:
    n_filters = 1 if params.depthwise else params.out_channels
    im_channels = 1 if params.depthwise else params.in_channels
    kernel_size = params.kernel_size
    patch_size = params.patch_size

    y = torch.empty((patch_size, n_filters), dtype=x.dtype, requires_grad=False)

    for n_filter in range(n_filters):
        # We need HWIO, originally OIHW, so we need to make it HWI before flattening (so that memory layout becomes [H][W][I]
        y[:, n_filter:n_filter+1] = x[n_filter].permute(1,2,0).reshape(patch_size, 1)

    return y

def bias2col(x: torch.Tensor, params: ConvParams) -> torch.Tensor:
    # TODO This can just be a one-dimensional bias, and then we can read it with a zero-byte increment
    
    n_patches = params.n_patches
    out_channels = params.out_channels

    y = torch.empty((n_patches, out_channels), dtype=x.dtype, requires_grad=False)
    
    for row in range(n_patches):
        y[row] = x

    return y

def im2col(x: torch.Tensor, params: ConvParams) -> torch.Tensor:
    """
    Expects input image in (N, C, H, W) format
    """
    assert x.shape[2] == x.shape[3]

    batch_size = params.batch_size
    im_channels = 1 if params.depthwise else params.in_channels
    im_height = params.in_row_dim
    im_width = params.in_col_dim
    kernel_size = params.kernel_size
    stride = params.stride
    padding = params.padding
    n_patches = params.n_patches
    patch_size = params.patch_size

    y = torch.empty((n_patches, patch_size), dtype=x.dtype, requires_grad=False)

    patch_row = 0

    for n_batch in range(batch_size):
        for im_row in range(-padding, im_height - kernel_size + padding + 1, stride):
            for im_col in range(-padding, im_width - kernel_size + padding + 1, stride):
                patch_col = 0

                for im_channel in range(im_channels):
                    for filter_row in range(kernel_size):
                        for filter_col in range(kernel_size):
                            pixel_row = im_row + filter_row
                            pixel_col = im_col + filter_col

                            if (pixel_row < 0 or pixel_row >= im_height
                                or pixel_col < 0 or pixel_col >= im_width):
                                y[patch_row][patch_col] = 0
                            else:
                                y[patch_row][patch_col] = x[n_batch][im_channel][pixel_row][pixel_col]

                            patch_col += 1
                patch_row += 1

    return y


def pool_with_col2im(x: torch.Tensor, params: ConvParams) -> torch.Tensor:
    assert(x.dim() == 2)
    batch_size = params.batch_size
    channels = params.out_channels
    in_row_dim = params.out_row_dim
    in_col_dim = params.out_col_dim
    kernel_size = params.pool_size
    stride = params.pool_stride
    padding = params.pool_padding
    # n_patches = params.n_patches
    # patch_size = params.patch_size 
    out_row_dim = params.out_dim_pooled
    out_col_dim = params.out_dim_pooled

    y = torch.empty((batch_size, out_row_dim, out_col_dim, channels), dtype=x.dtype, requires_grad=False)

    for batch in range(batch_size):
        for channel in range(channels):
            for out_row in range(out_row_dim):
                for out_col in range(out_col_dim):
                    in_row = out_row * stride - padding
                    pooled = -9999
                    for kernel_row in range(kernel_size):
                        in_col = out_col * stride - padding
                        for kernel_col in range(kernel_size):
                            if in_row >= 0 and in_row < in_row_dim and in_col >= 0 and in_col < in_col_dim:
                                if x[batch*in_row_dim*in_col_dim + in_row*in_col_dim + in_col][channel] > pooled:
                                    pooled = x[batch*in_row_dim*in_col_dim + in_row*in_col_dim + in_col][channel]
                            else:
                                if 0 > pooled:
                                    pooled = 0
                            in_col += 1
                        in_row += 1
                    y[batch][out_row][out_col][channel] = pooled

    return y