try:
    import _init_paths   # pylint: disable=unused-import
except:
    pass
from datasets import get_test_data
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torchvision.utils
from contextlib import redirect_stdout
import torchvision.transforms as transforms
from tools.model_quantization import quantize_model, profile_model
from models.nn_non_decomposed import MultiConceptNonDecomposed
from models.vsa import get_vsa
from torch import nn
import sys
import os
import json
from config import *
from tools.gemmini import *
from math import ceil, log2
from tqdm import tqdm
from typing import Tuple
from models.resnet import Bottleneck
import argparse


def imshow(img):
    # img = img / 2 + 0.5     # unnormalize (from [-1,1] to [0,1])
    plt.imshow(img.permute(1,2,0))
    plt.show()

def generate_image_png(images: Tensor, dir):
    """
    images: Tensor of shape (N, C, H, W).
    """
    os.makedirs(dir, exist_ok=True)
    for i, img in enumerate(tqdm(images, desc="Generating images", leave=False)):
        torchvision.utils.save_image(img, f"{dir}/image_{i}.png")

def print_tensor(t: torch.Tensor, decimals = 5):
    if len(t.shape) == 0:
        print(round(t.item(), decimals), end="")
        return
    
    print("{", end="")
    
    for i,m in enumerate(t):
        print_tensor(m)

        if i < len(t) - 1:
            print(",", end="")

    print("}", end="")

def generate_image_header(images: Tensor, scaling_factor = None):
    """
    images: Tensor of shape (N, C, H, W)
    """
    
    print(r'''#ifndef MC_MNIST_IMAGES_H
#define MC_MNIST_IMAGES_H

#include "gemmini/gemmini_params.h"
''')

    # Gemmini expects (N, H, W, C)
    _images = images.permute(0, 2, 3, 1)

    shape = _images.shape

    print("#define NUM_IMAGES {}".format(shape[0]))
    print(f"static const elem_t images[NUM_IMAGES][{shape[1]}][{shape[2]}][{shape[3]}] row_align(1) = ", end="")
    if scaling_factor is not None:
        _images = torch.clamp(_images * scaling_factor, min=-128, max=127).round().int()
    print_tensor(_images)

    print(";\n\n#endif\n")

def generate_codebook_header(vsa):
    print(f'''#ifndef MC_MNIST_CODEBOOKS_H
#define MC_MNIST_CODEBOOKS_H

#include <stdint.h>

#define NUM_CODEBOOKS {len(vsa.codebooks)}
''')

    for i, codebook in enumerate(tqdm(vsa.codebooks, desc="Generating codebooks", leave=False)):
        if vsa.mode == "HARDWARE":
            _codebook = codebook[:, :vsa.fold_dim]
        else:
            _codebook = codebook
        shape = _codebook.shape
        print(f"static const uint8_t codebook_{i}[{shape[0]}][{shape[1]}] = ", end="")
        print_tensor(_codebook)
        print(";")

    print("\n\n#endif\n")

def generate_target_header(targets: Tensor):
    print(r'''#ifndef MC_MNIST_TARGETS_H
#define MC_MNIST_TARGETS_H

#include <stdint.h>
''')
    print(f"static const int8_t target_vectors[{targets.size(0)}][{targets.size(1)}] = ", end="")
    print_tensor(targets)
    print(";\n\n#endif\n")


def generate_labels(labels: list, filename):
    """
    Dump out labels as a json file
    """
    # Convert to dict
    label_set = []
    for label in tqdm(labels, desc="Generating labels", leave=False):
        d = []
        for obj in label:
            d.append(
                {
                    'pos_x': obj[0],
                    'pos_y': obj[1],
                    'color': obj[2],
                    'digit': obj[3]
                }
            )
        label_set.append(d)

    with open(filename, "w") as f:
        json.dump(label_set, f)

def generate_model_params(model, gemmini_dim, batch_size = 1, decimals = 5):
    """
    Dump out model as a header file
    This function takes a unquantized model (for now)
    """
    def print_matrix2d(m: torch.Tensor, height: int, width: int, height_padded: int, width_padded: int):
        print("{", end="")
        for i in range(height_padded):
            print("{", end="")

            for j in range(width_padded):
                if j == width_padded - 1:
                    end = ""
                else:
                    end = ","

                if i < height and j < width:
                    print(round(m[i][j].item(), decimals), end=end)
                else:
                    print("0", end=end)

            if i == height_padded - 1:
                end = ""
            else:
                end = ","
            print("}", end=end)

        print("}", end="")

    def padded(dim: int) -> int:
        return dim
        # return int(ceil(dim / gemmini_dim)) * gemmini_dim

    def conv_header(layer: nn.Conv2d, idx: int, pool_size: int, pool_stride: int, pool_padding: int, relu: bool):
        input_shape = layer.input_shape

        if layer.groups > 1:
            return conv_dw_header(layer, idx, pool_size=pool_size, pool_stride=pool_stride, pool_padding=pool_padding, relu=relu)

        params = get_params(layer, in_dim=input_shape[-1], pool_stride=pool_stride, pool_size=pool_size, pool_padding=pool_padding, batch_size=batch_size)

        # Print the weight matrix
        w = filter2col(layer.weight, params)

        w_height_padded = padded(w.shape[0])
        w_width_padded = padded(w.shape[1])

        print("static const elem_t conv_{}_w[{}][{}] row_align(1) = ".format(idx, w_height_padded, w_width_padded), end="")
        print_matrix2d(w, w.shape[0], w.shape[1], w_height_padded, w_width_padded)
        print(";")

        # Print the bias matrix
        if params.bias:
            # b = bias2col(layer.bias, params)
            b = layer.bias

            # b_height_padded = padded(b.shape[0])
            # b_width_padded = padded(b.shape[1])
            b_width_padded = padded(b.shape[0])

            # print("static const acc_t conv_{}_b[{}][{}] row_align_acc(1) = ".format(idx, b_height_padded, b_width_padded), end="")
            # print_matrix2d(b, b.shape[0], b.shape[1], b_height_padded, b_width_padded)
            # print(";")
            print("static const acc_t conv_{}_b[{}] row_align_acc(1) = {{".format(idx, b_width_padded), end="")
            for i in range(b_width_padded):
                elem = round(b[i].item(), decimals) if i < b.shape[0] else 0
                end = "," if i < b_width_padded-1 else "};\n"
                print("{}".format(elem), end=end)

        # Print the input matrix (if im2col is necessary)
        in_height = params.n_patches
        in_width = params.patch_size

        print("static elem_t conv_{}_in[{}][{}] row_align(1);".format(idx, padded(in_height), padded(in_width)))

        # Print the output matrix
        out_height = in_height
        out_width = w.shape[1]

        print("static elem_t conv_{}_out[{}][{}] row_align(1);".format(idx, padded(out_height), padded(out_width)))

        if params.is_pooled:
            print(f"static elem_t conv_{idx}_out_pooled[{params.batch_size}][{params.out_dim_pooled}][{params.out_dim_pooled}][{params.out_channels}];")

        # Print params
        # print(f"static const struct ConvParams conv_{idx}_params = {{.batch_size={params.batch_size}, .in_row_dim={params.in_row_dim}, .in_col_dim={params.in_col_dim}, .kernel_size={params.kernel_size}, .in_channels={params.in_channels}, .out_channels={params.out_channels}, .stride={params.stride}, .padding={params.padding}, .bias={int(params.bias)}, .depthwise={int(params.depthwise)}, .out_row_dim={params.out_row_dim}, .out_col_dim={params.out_col_dim}, .n_patches={params.n_patches}, .patch_size={params.patch_size}, .pool_size={pool_size}, .pool_stride={pool_stride}, .pool_padding={pool_padding}, .out_dim_pooled={params.out_dim_pooled}, .output_scale={-int(log2(params.output_scale))}, .I={padded(in_height)}, .J={padded(out_width)}, .K={padded(in_width)}, .res_scale={-int(log2(res_scale))}}};")
        print(f"static const struct ConvParams conv_{idx}_params = {{.batch_size={params.batch_size}, .in_row_dim={params.in_row_dim}, .in_col_dim={params.in_col_dim}, .out_row_dim={params.out_row_dim}, .out_col_dim={params.out_col_dim}, .kernel_size={params.kernel_size}, .in_channels={params.in_channels}, .out_channels={params.out_channels}, .stride={params.stride}, .padding={params.padding}, .bias={int(params.bias)}, .depthwise={int(params.depthwise)}, .n_patches={params.n_patches}, .patch_size={params.patch_size}, .pool_size={pool_size}, .pool_stride={pool_stride}, .pool_padding={pool_padding}, .out_dim_pooled={params.out_dim_pooled}, .output_scale={params.output_scale}, .I={padded(in_height)}, .J={padded(out_width)}, .K={padded(in_width)}, .res_scale={res_scale}}};")
        layer.quant_params = params
        layer.quant_params.I = padded(in_height)
        layer.quant_params.J = padded(out_width)
        layer.quant_params.K = padded(in_width)
        layer.quant_params.relu = relu
        layer.quant_params.idx = idx

        return (params.batch_size,
                params.out_channels,
                params.out_row_dim,
                params.out_col_dim)

    def conv_dw_header(layer: nn.Conv2d, idx: int, pool_size: int, pool_stride: int, pool_padding: int, relu: bool):

        input_shape = layer.input_shape

        params = get_params(layer, in_dim=input_shape[-1], pool_stride=pool_stride, pool_size=pool_size, pool_padding=pool_padding, batch_size=batch_size)

        # TODO This runs on the CPU for now

        # Print the weight matrices # TODO the channels should be the innermost layer here as well
        print("static const elem_t conv_dw_{}_w[{}][{}][{}] row_align(1) = {{".format(idx, params.in_channels, params.kernel_size, params.kernel_size), end="")
        for i in range(params.in_channels):
            print_matrix2d(layer.weight[i][0], layer.weight[i][0].shape[0], layer.weight[i][0].shape[1], layer.weight[i][0].shape[0], layer.weight[i][0].shape[1])
            if i < params.in_channels - 1:
                print(",", end="")
        print("};")

        # Print the biases
        if params.bias:
            print("static const acc_t conv_dw_{}_b[{}] row_align_acc(1) = {{".format(idx, params.in_channels), end="")
            for i in range(params.in_channels):
                if i == params.in_channels-1:
                    end = ""
                else:
                    end = ","

                print(layer.bias[i].int().item(), end=end)
            print("};")

        # Print the output
        # print("static elem_t conv_dw_{}_out[{}][{}][{}][{}] row_align(1);".format(idx, params.batch_size, params.out_row_dim, params.out_col_dim, params.out_channels))
        print("static elem_t conv_dw_{}_out[{}][{}] row_align(1);".format(idx, padded(params.batch_size * params.out_row_dim * params.out_col_dim), padded(params.out_channels)))

        if params.is_pooled:
            print(f"static elem_t conv_dw_{idx}_out_pooled[{params.batch_size}][{params.out_channels}][{params.out_row_dim_pooled}][{params.out_col_dim_pooled}];")

        # Print params
        print(f"static const struct ConvParams conv_dw_{idx}_params = {{.batch_size={params.batch_size}, .in_row_dim={params.in_row_dim}, .in_col_dim={params.in_col_dim}, .kernel_size={params.kernel_size}, .in_channels={params.in_channels}, .out_channels={params.out_channels}, .stride={params.stride}, .padding={params.padding}, .bias={int(params.bias)}, .depthwise={int(params.depthwise)}, .out_row_dim={params.out_row_dim}, .out_col_dim={params.out_col_dim}, .n_patches={params.n_patches}, .patch_size={params.patch_size}, .pool_size={pool_size}, .pool_stride={pool_stride}, .pool_padding={pool_padding}, .out_dim_pooled={params.out_dim_pooled}, .output_scale={params.output_scale}, .res_scale={res_scale}, .I={padded(params.batch_size * params.out_row_dim * params.out_col_dim)}, .J={padded(params.out_channels)}}};")
        # print(f"static const struct ConvParams conv_dw_{idx}_params = {{.batch_size={params.batch_size}, .in_row_dim={params.in_row_dim}, .in_col_dim={params.in_col_dim}, .kernel_size={params.kernel_size}, .in_channels={params.in_channels}, .out_channels={params.out_channels}, .stride={params.stride}, .padding={params.padding}, .bias={int(params.bias)}, .depthwise={int(params.depthwise)}, .out_row_dim={params.out_row_dim}, .out_col_dim={params.out_col_dim}, .n_patches={params.n_patches}, .patch_size={params.patch_size}, .pool_size={pool_size}, .pool_stride={pool_stride}, .pool_padding={pool_padding}, .out_dim_pooled={params.out_dim_pooled}, .output_scale={-int(log2(params.output_scale))}, .res_scale={-int(log2(res_scale))}, .I={padded(params.batch_size * params.out_row_dim * params.out_col_dim)}, .J={padded(params.out_channels)}}};")

        layer.quant_params = params
        layer.quant_params.relu = relu
        layer.quant_params.idx = idx

        return (params.batch_size, params.out_channels, params.out_row_dim, params.out_col_dim)

    def fc_header(layer: nn.Conv2d, idx: int, relu: bool):
        input_shape = layer.input_shape

        params = get_params(layer, batch_size=batch_size)

        # Print weights
        w = layer.weight.permute(1, 0)
        print("static const elem_t fc_{}_w[{}][{}] row_align(1) = ".format(idx, padded(w.shape[0]), padded(w.shape[1])), end="")
        print_matrix2d(w, w.shape[0], w.shape[1], padded(w.shape[0]), padded(w.shape[1]))
        print(";")

        if params.bias:
            # Print bias
            print("static const acc_t fc_{}_b[{}][{}] row_align_acc(1) = ".format(idx, padded(params.batch_size), padded(layer.bias.shape[0])), end="")

            print("{", end="")

            for j in range(padded(params.batch_size)):
                print("{", end="")
                for i in range(padded(layer.bias.shape[0])):
                    if i == padded(layer.bias.shape[0]) - 1:
                        end = ""
                    else:
                        end = ","

                    if i < layer.bias.shape[0] and j < params.batch_size:
                        print(round(layer.bias[i].item(), decimals), end=end)
                    else:
                        print("0", end=end)

                if j == padded(params.batch_size) - 1:
                    end = ""
                else:
                    end = ","

                print("}", end=end)

            print("};")

        # Print output
        print("static elem_t fc_{}_out[{}][{}] row_align(1);".format(idx, padded(params.batch_size), padded(params.out_features)))

        # Print params
        # print(f"static const struct FcParams fc_{idx}_params = {{.batch_size={params.batch_size}, .in_features={params.in_features}, .out_features={params.out_features}, .bias={int(params.bias)}, .output_scale={-int(log2(params.output_scale))}, .I={padded(params.out_features)}, .J={padded(params.batch_size)}, .K={padded(layer.weight.shape[1])}}};")
        print(f"static const struct FcParams fc_{idx}_params = {{.batch_size={params.batch_size}, .in_features={params.in_features}, .out_features={params.out_features}, .bias={int(params.bias)}, .output_scale={params.output_scale}, .I={padded(params.batch_size)}, .J={padded(params.out_features)}, .K={padded(params.in_features)}}};")

        layer.quant_params = params
        layer.quant_params.I = padded(params.batch_size)
        layer.quant_params.J = padded(params.out_features)
        layer.quant_params.K = padded(params.in_features)
        layer.quant_params.relu = relu
        layer.quant_params.idx = idx

        return (params.batch_size, params.out_features)

    def is_follow_on_module(mod):
        for f in follow_on_modules:
            if isinstance(mod, f):
                return True
        return False

    print('''#ifndef MC_MNIST_MODEL_PARAMS_H
#define MC_MNIST_MODEL_PARAMS_H
''')

    print(f'''#include "gemmini/gemmini_params.h"
#include "gemmini/gemmini.h"
#include "gemmini/gemmini_nn.h"
#include <stdbool.h>

#define BATCH_SIZE {batch_size}

''')

    idx = 1
    res_scale = 1

    inner_modules = list(model.modules())
    for i in tqdm(range(len(inner_modules)), desc="Generating model params header", leave=False):
        layer = inner_modules[i]

        # if isinstance(layer, models.resnet.Bottleneck) or isinstance(layer, torchvision.models.resnet.BasicBlock):
        #     res_scale = layer.add_scale

        if isinstance(layer, nn.Conv2d):
            pool_size = 1
            pool_stride = 1
            pool_padding = 0
            relu = False

            follow_on_modules = [
                nn.modules.activation.ReLU6,
                nn.modules.activation.ReLU,
                nn.modules.batchnorm.BatchNorm2d,
                nn.modules.pooling.MaxPool2d
            ] 

            j = i + 1
            while j < len(inner_modules) and (is_follow_on_module(inner_modules[j]) or len(list(inner_modules[j].children())) != 0):
                if isinstance(inner_modules[j], nn.modules.pooling.MaxPool2d):
                    pool_size = inner_modules[j].kernel_size
                    pool_stride = inner_modules[j].stride
                    pool_padding = inner_modules[j].padding
                elif isinstance(inner_modules[j], nn.modules.activation.ReLU) or isinstance(inner_modules[j], nn.modules.activation.ReLU6):
                    relu = True

                j += 1
            conv_header(layer, idx, pool_size=pool_size, pool_stride=pool_stride, pool_padding=pool_padding, relu=relu)
            print("\n")
            idx += 1

        elif isinstance(layer, nn.Linear):
            relu = False

            follow_on_modules = [
                nn.modules.activation.ReLU6,
                nn.modules.activation.ReLU,
            ]

            def is_follow_on_module(mod):
                for f in follow_on_modules:
                    if isinstance(mod, f):
                        return True
                return False

            j = i + 1
            while j < len(inner_modules) and (is_follow_on_module(inner_modules[j]) or len(list(inner_modules[j].children())) != 0):
                if isinstance(inner_modules[j], nn.modules.activation.ReLU) or isinstance(inner_modules[j], nn.modules.activation.ReLU6):
                    relu = True

                j += 1

            fc_header(layer, idx, relu)
            print("\n")
            idx += 1

    print("#endif\n")


def generate_model_body(model: nn.Module):
    def name_of_output(params):
        if isinstance(params, ConvParams):
            if params.depthwise:
                if params.is_pooled:
                    return f"conv_dw_{params.idx}_out_pooled"
                else:
                    return f"conv_dw_{params.idx}_out"
            else:
                if params.is_pooled:
                    return f"conv_{params.idx}_out_pooled"
                else:
                    return f"conv_{params.idx}_out"
        elif isinstance(params, FcParams):
            return f"fc_{params.idx}_out"
        elif params == "images":
            return "images"
        else:
            raise Exception("Unknown output")

    def name_of_params(params):
        if isinstance(params, ConvParams):
            return f"conv_{params.idx}_params"
        elif isinstance(params, FcParams):
            return f"fc_{params.idx}_params"
        elif params == "images":
            return "ERROR"
        else:
            raise Exception("Unknown params")

    def is_col2imed(params):
        return params == "images" or (isinstance(params, ConvParams) and params.is_pooled)

    def conv_action(layer: nn.Conv2d, last) -> Tuple[str, str]:
        if layer.groups > 1:
            return conv_dw_action(layer, last)

        idx = layer.quant_params.idx
        relu = layer.quant_params.relu

        layer_name = f"conv_{idx}"
        params_name = f"{layer_name}_params"
        bias = "NULL" if layer.bias is None else f"{layer_name}_b"
        activation = "RELU" if relu else "NO_ACTIVATION"

        last_params_name = name_of_params(last)
        last_output_name = name_of_output(last)

        col2im_needed = not is_col2imed(last) # isinstance(last, ConvParams) and not last.is_pooled # and not last.depthwise

        params = layer.quant_params

        layer_input_name = f"{layer_name}_in"

        print(f'    // {layer_name}')

        if not col2im_needed:
            print(f'''    start = read_cycles();
    
    im2col({params_name}.batch_size, {params_name}.in_channels, 
        {params_name}.in_row_dim, {params_name}.in_col_dim,
        {params_name}.I, {params_name}.K,
        {last_output_name}, {layer_name}_in, &{params_name});
    
    end = read_cycles();
    im2col_cycles += end - start;
''')
        elif params.kernel_size > 1 or params.stride > 1 or params.padding > 1:
            print(f'''    start = read_cycles();

    im2col_with_col2im({last_params_name}.I, {last_params_name}.J,
        {params_name}.I, {params_name}.K,
        {last_output_name}, {layer_name}_in, &{params_name});

    end = read_cycles();
    im2col_cycles += end - start;
''')
        else:
            layer_input_name = last_output_name

        print(f'''    start = read_cycles();

    tiled_matmul_nn_auto({params_name}.I, {params_name}.J, {params_name}.K,
        {layer_input_name}, {layer_name}_w, {bias}, {layer_name}_out,
        {activation}, {params_name}.output_scale, true,
        tiled_matmul_type, check, "{layer_name}");

    end = read_cycles();
    matmul_cycles += end - start;
''')

        if params.is_pooled:
            print(f'''    start = read_cycles();

    pool_with_col2im({params_name}.I, {params_name}.J,
        {params_name}.batch_size, {params_name}.out_channels,
        {params_name}.out_dim_pooled, {params_name}.out_dim_pooled,
        {layer_name}_out, {layer_name}_out_pooled, &{params_name});

    end = read_cycles();
    pool_cycles += end - start;
''')

        return params

    def fc_action(layer: nn.Linear, last) -> str:
        relu = layer.quant_params.relu
        idx = layer.quant_params.idx

        layer_name = f"fc_{idx}"
        params_name = f"{layer_name}_params"
        last_params_name = name_of_params(last)
        last_output_name = name_of_output(last)
        bias = "NULL" if layer.bias is None else f"{layer_name}_b"
        activation = "RELU" if relu else "NO_ACTIVATION"

        params = layer.quant_params

        col2im_needed = not is_col2imed(last) # isinstance(last, ConvParams) and not last.is_pooled # and not last.depthwise
        # TODO: this identifies the final fc layer in resnet, which is followed by a avgpool. We need a different way to handle this, maybe add other avgpool action
        if isinstance(last, ConvParams) and (isinstance(model, MultiConceptNonDecomposed)):
            print(f'''    // Global averaging
    static elem_t average[{params.I}][{params.K}] row_align(1);

    start = read_cycles();
    tiled_global_average_auto({last_output_name}, average, {last_params_name}.batch_size,
        {last_params_name}.out_channels, {last_params_name}.out_row_dim, tiled_matmul_type);
    
    end = read_cycles();
    other_cycles += end - start;
''')

            last_output_name = "average"

        # TODO: looks like this is for models where There's no avgpool between Conv and FC. Will comment it out for now.
#         elif isinstance(last, ConvParams) and not col2im_needed:
#             print(f'''    // Convert conv output to fc input
#     static elem_t {layer_name}_in[{params.K}][{params.J}] row_align(1);

#     start = read_cycles();

#     for (size_t batch = 0; batch < {last_params_name}.batch_size; batch++) {{
#         size_t pixel = 0;
#         for (size_t channel = 0; channel < {last_params_name}.out_channels; channel++) {{
#             for (size_t row = 0; row < {last_params_name}.{"out_dim_pooled" if last.is_pooled else "out_dim"}; row++) {{
#                 for (size_t col = 0; col < {last_params_name}.{"out_dim_pooled" if last.is_pooled else "out_dim"}; col++) {{
#                     {layer_name}_in[pixel][batch] = {last_output_name}[batch][row][col][channel];
#                     pixel++;
#                 }}
#             }}
#         }}
#     }}

#     end = read_cycles();
#     other_cycles += end - start;
# ''')
#             last_output_name = f"{layer_name}_in"

        print(f'''    // {layer_name}
    start = read_cycles();

    tiled_matmul_nn_auto({params_name}.I, {params_name}.J, {params_name}.K,
        {last_output_name}, {layer_name}_w, {bias}, {layer_name}_out,
        {activation}, {params_name}.output_scale, false,
        tiled_matmul_type, check, "{layer_name}");

    end = read_cycles();
    matmul_cycles += end - start;
''')

        params = layer.quant_params
        return params

    def conv_dw_action(layer: nn.Conv2d, last) -> Tuple[str, str]:
        relu = layer.quant_params.relu
        idx = layer.quant_params.idx

        layer_name = f"conv_dw_{idx}"
        params_name = f"{layer_name}_params"
        bias = "NULL" if layer.bias is None else f"{layer_name}_b"
        activation = "RELU" if relu else "NO_ACTIVATION"

        last_output_name = name_of_output(last)
        last_params_name = name_of_params(last)

        col2im_needed = not is_col2imed(last) # isinstance(last, ConvParams) and not last.is_pooled # and not last.depthwise

        if col2im_needed:
            print(f'''    // {layer_name}
    start = read_cycles();

    conv_dw_with_col2im({last_params_name}.I, {last_params_name}.J, {params_name}.I, {params_name}.J,
        {params_name}.batch_size, {params_name}.in_channels, {params_name}.out_dim, {params_name}.kernel_size,
        {last_output_name}, {layer_name}_w, {bias}, {layer_name}_out, &{params_name});

    end = read_cycles();
    conv_dw_cycles += end - start;
''')
        else:
            print(f'''    // {layer_name}
    start = read_cycles();

    conv_dw({params_name}.I, {params_name}.J
        {params_name}.batch_size, {params_name}.in_channels, 
        {params_name}.in_row_dim, {params_name}.in_col_dim,
        {params_name}.out_row_dim, {params_name}.out_col_dim,
        {params_name}.kernel_size,
        {last_output_name}, {layer_name}_w, {bias}, {layer_name}_out, &{params_name});

    end = read_cycles();
    conv_dw_cycles += end - start;
''')

        params = layer.quant_params

        if params.is_pooled:
            print(f'''
    start = read_cycles();

    pool_with_col2im({params_name}.I, {params_name}.J,
        {params_name}.batch_size, {params_name}.out_channels, {params_name}.out_dim_pooled,
        {layer_name}_out, {layer_name}_out_pooled, &{params_name});

    end = read_cycles();
    pool_cycles += end - start;
''')

        return params

    def res_add_action(res_input, last):
        initial_input = name_of_output(res_input)
        last_output = name_of_output(last)

        initial_params_name = name_of_params(res_input)
        last_params_name = name_of_params(last)

        relu = "true" # ! Our model so far always has relu after res add (the conv2 revolved in res_add should have relu set to false)

        print(f'''    // Add residuals
    start = read_cycles();
    tiled_resadd_auto({last_params_name}.I, {last_params_name}.J,
        {last_params_name}.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        {initial_input},
        {last_output},
        {last_output},
        {relu},
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    ''')

        return last

    def res_downsample_action(layer: nn.Conv2d, initial):
        # This function is only used for ResNet right now
        print(f"    // Downsampling {name_of_output(initial)}")
        downsampled = conv_action(layer, initial)
        return downsampled

    ## Function body starts here
    # First, add the code to compute the features
    last = "images"

    res_input = None # Used for ResNet
    res_add_layer = None
    res_downsample = None # Used for ResNet

    final_fc = [mod for mod in model.modules() if isinstance(mod, nn.Linear)][-1]
    
    print(r'''#ifndef MC_MNIST_MODEL_H
#define MC_MNIST_MODEL_H
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "gemmini/gemmini.h"
#include "gemmini/gemmini_nn.h"

#include "model_params.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
''')

    print(f"typedef elem_t fc_t[BATCH_SIZE][{final_fc.quant_params.out_features}];")

    print(r'''static fc_t *model(const elem_t *images, enum tiled_matmul_type_t tiled_matmul_type, bool check){
    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;

    gemmini_flush(0);
''')

    for layer in tqdm(model.modules(), desc="Generating model header", leave=False):

        if layer == res_downsample:
            res_input = res_downsample_action(layer, res_input)

        elif isinstance(layer, Bottleneck) or isinstance(layer, torchvision.models.resnet.BasicBlock):
            res_input = last
            res_add_layer = list(layer.modules())[-1]

            if layer.downsample is not None:
                res_downsample = layer.downsample[0]
            else:
                res_downsample = None

        elif isinstance(layer, nn.Conv2d):
            last = conv_action(layer, last)

        elif isinstance(layer, nn.Linear):
            last = fc_action(layer, last)

        if res_add_layer == layer:   # last layer in a residual block
            last = res_add_action(res_input, last) # res_input should already be pointed to the correct input, either downsampled or not

    last_params = name_of_params(last)
    last_output = name_of_output(last)

    print(f'''uint64_t total_cycles = im2col_cycles + matmul_cycles + pool_cycles + conv_dw_cycles + res_add_cycles + other_cycles;

    printf("\\nTotal cycles: %lu (100%%)\\n", total_cycles);
    printf("Matmul cycles: %lu (%ld%%)\\n", matmul_cycles, (matmul_cycles * 100) / total_cycles);
    printf("Im2col cycles: %lu (%ld%%)\\n", im2col_cycles, (im2col_cycles * 100) / total_cycles);
    printf("Conv cycles: %lu (%ld%%)\\n", conv_cycles, (conv_cycles * 100) / total_cycles);
    printf("Pooling cycles: %lu (%ld%%)\\n", pool_cycles, (pool_cycles * 100) / total_cycles);
    printf("Depthwise convolution cycles: %lu (%ld%%)\\n", conv_dw_cycles, (conv_dw_cycles * 100) / total_cycles);
    printf("Res add cycles: %lu (%ld%%)\\n", res_add_cycles, (res_add_cycles * 100) / total_cycles);
    printf("Other cycles: %lu (%ld%%)\\n", other_cycles, (other_cycles * 100) / total_cycles);

    return &{last_output};
''')
    print('''}

#pragma GCC diagnostic pop
#endif
''')

#     print(r'''
# int main (int argc, char * argv[]) {
# #ifndef BAREMETAL
#     if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
#       perror("mlockall failed");
#       exit(1);
#     }
# #endif

#     enum tiled_matmul_type_t tiled_matmul_type;
#     if (argc < 2) {
#         tiled_matmul_type = WS;
#     } else if (strcmp(argv[1], "cpu") == 0) {
#         tiled_matmul_type = CPU;
#     } else if (strcmp(argv[1], "os") == 0) {
#         tiled_matmul_type = OS;
#     } else if (strcmp(argv[1], "ws") == 0) {
#         tiled_matmul_type = WS;
#     } else if (strcmp(argv[1], "-h") == 0) {
#         printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
#         exit(0);
#     } else {
#         printf("Unknown command-line argument\n");
#         printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
#         exit(1);
#     }

#     bool check;
#     if (argc < 3) {
#         check = false;
#     } else if (strcmp(argv[2], "check") == 0) {
#         check = true;
#     } else {
#         printf("Unknown command-line argument\n");
#         printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
#         exit(1);
#     }

#     model(tiled_matmul_type, false, check);
# ''')

#     print(f'''    // Find highest probs
#     for (int batch = 0; batch < {batch_size}; batch++) {{
#         elem_t max_prob = {last_output}[0][batch];
#         size_t max_idx = 0;

#         for (int i = 1; i < {last_params}.out_features; i++) {{
#             if ({last_output}[i][batch] > max_prob) {{
#                 max_prob = {last_output}[i][batch];
#                 max_idx = i;
#             }}
#         }}
        
#         printf("Prediction: %u (score: %d)\\n", max_idx, max_prob);
#     }}
# ''')

#    print("    return(0);\n}\n")

if __name__ == "__main__":
    test_dir = f"./tests/{VSA_MODE}-{DIM}dim{'-' + str(FOLD_DIM) + 'fd' if VSA_MODE=='HARDWARE' else ''}-{NUM_POS_X}x-{NUM_POS_Y}y-{NUM_COLOR}color/{ALGO}"
    parser = argparse.ArgumentParser(description="Dump model and test data to C")
    parser.add_argument("checkpoint", type=str, help="model checkpoint")
    parser.add_argument("--codebooks", type=str, help="codebook file path", default=None)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiConceptNonDecomposed(dim=DIM, device=device)
    model.eval()

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print("Invalid model checkpoint path.")
        exit(1)

    vsa = get_vsa(test_dir, VSA_MODE, ALGO, args.codebooks, DIM, MAX_NUM_OBJECTS, NUM_COLOR, NUM_POS_X, NUM_POS_Y, FOLD_DIM, EHD_BITS, SIM_BITS, SEED, device)
    dl = get_test_data(test_dir, vsa, False, NUM_TEST_SAMPLES, MAX_NUM_OBJECTS, SINGLE_COUNT, 1, NUM_POS_X, NUM_POS_Y, NUM_COLOR)
    quan_dl = get_test_data(test_dir, vsa, True, NUM_TEST_SAMPLES, MAX_NUM_OBJECTS, SINGLE_COUNT, 1, NUM_POS_X, NUM_POS_Y, NUM_COLOR)

    # TODO Can enable transform in dataset and pass this as a parameter, but must make sure the data to be loaded isn't too large (e.g. training ds) as it's costly for memory
    transform = transforms.Compose([
            # transforms.Resize(224, antialiased = True),
            transforms.ConvertImageDtype(torch.float32)  # Converts to [0, 1]
        ])

    images = transform(dl.dataset.data)

    dump_dir = "./dump"
    os.makedirs(dump_dir, exist_ok=True)

    print("Quantizing/Profiling model...")
    # quantize_model(model, quan_dl)

    # If not quantize, still need to run profiling to get the input shape of each layer
    profile_model(model, quan_dl)

    # for i, v in enumerate(x.permute(0, 2, 3, 1)):
    #     for j, q in enumerate(v):
    #         for k, p in enumerate(q):
    #             for (o, w) in enumerate(p):
    #                 print(f"[{i}][{j}][{k}][{o}] =", "{:.3f}".format(w.item()), end="\t")
    #             print()

    # for i, v in enumerate(im):
    #     for j, q in enumerate(v):
    #         print(f"[{i}][{j}] =", "{:.3f}".format(q.item()), end="\t")
    #     print()

    # Dump model headers
    with open(dump_dir + '/model_params.h', 'w') as f:
        with redirect_stdout(f):
            generate_model_params(model, batch_size=1, gemmini_dim=GEMMINI_DIM)

    with open(dump_dir + '/model.h', 'w') as f:
        with redirect_stdout(f):
            generate_model_body(model)

    # Dump codebooks
    with open(dump_dir + '/codebooks.h', 'w') as f:
        with redirect_stdout(f):
            generate_codebook_header(vsa)
    
    # Dump targets
    # No easy way to read in json file in baremetal form, so we'll dump out the ground-truth vector for each test
    with open(dump_dir + "/targets.h", 'w') as f:
        with redirect_stdout(f):
            generate_target_header(dl.dataset.targets)

    # Dump test image header and PNGs
    with open(dump_dir + '/images.h', 'w') as f:
        with redirect_stdout(f):
            generate_image_header(images)
    generate_image_png(images, dump_dir + '/images')

    # Dump test labels
    generate_labels(dl.dataset.labels, dump_dir + '/labels.json')

    print("[ Done ]")
