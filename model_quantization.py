import numpy as np
from main import DIM, get_test_data, get_vsa
from model.nn_non_decomposed import MultiConceptNonDecomposed
import torch
import torchvision
from typing import List, Tuple
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model():
    model = MultiConceptNonDecomposed(dim=DIM, device=device)
    return model

def flattened_children(model: nn.Module) -> List[nn.Module]:
    result = []
    for layer in model.modules():
        if len(list(layer.children())) == 0:
            result.append(layer)
    return result

def closest_lower_power_of_2(x: float) -> float:
    result = 1.0
    
    if x >= 1.0:
        next = lambda x: x * 2.0
        finished = lambda : next(result) > x
    else:
        next = lambda x: x / 2.0
        finished = lambda : next(result) < x
        
    while not finished():
        result = next(result)
        
    return result

def merge_biases(model):
    # Stolen from https://github.com/pytorch/pytorch/pull/901/files
    def absorb_bn(module, bn_module):
        w = module.weight.data
        if module.bias is None:
            zeros = torch.Tensor(module.out_channels).zero_().type(w.type())
            module.bias = nn.Parameter(zeros)
        b = module.bias.data
        invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
        w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
        b.add_(-bn_module.running_mean).mul_(invstd)

        if bn_module.affine:
            w.mul_(bn_module.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
            b.mul_(bn_module.weight.data).add_(bn_module.bias.data)

    children = flattened_children(model)
    for conv, bn in zip(children, children[1:]):
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            absorb_bn(conv, bn)

def remove_biases(model):
    for child in flattened_children(model):
        if isinstance(child, nn.BatchNorm2d):
            child.eps = 0
            child.running_mean.data = torch.zeros(child.running_mean.shape[0]).to(device)
            child.running_var.data = torch.ones(child.running_var.shape[0]).to(device)
            child.weight.data = torch.ones(child.weight.shape[0]).to(device)
            child.bias.data = torch.zeros(child.bias.shape[0]).to(device)

def quantized_weights(weights: torch.Tensor) -> Tuple[torch.Tensor, float]:
    '''
    Quantize the weights so that all values are integers between -128 and 127.
    You may want to use the total range, 3-sigma range, or some other range when
    deciding just what factors to scale the float32 values by.

    Parameters:
    weights (Tensor): The unquantized weights

    Returns:
    (Tensor, float): A tuple with the following elements:
                        * The weights in quantized form, where every value is an integer between -128 and 127.
                          The "dtype" will still be "float", but the values themselves should all be integers.
                        * The scaling factor that your weights were multiplied by.
                          This value does not need to be an 8-bit integer.
    '''

    max_ = weights.view(-1).max().item()
    min_ = weights.view(-1).min().item()

    max_mag = max(abs(max_), abs(min_))
    range = max_mag
    scale = 127.0 / range
    scale = closest_lower_power_of_2(scale)

    result = (weights * scale).round()
    return torch.clamp(result, min=-128, max=127), scale

def quantize_layer_weights(model: nn.Module):
    for layer in flattened_children(model):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            q_layer_data, scale = quantized_weights(layer.weight.data)
            q_layer_data = q_layer_data.to(device)

            layer.weight.data = q_layer_data
            layer.weight.scale = scale

            if (q_layer_data < -128).any() or (q_layer_data > 127).any():
                raise Exception("Quantized weights of {} layer include values out of bounds for an 8-bit signed integer".format(layer.__class__.__name__))
            if (q_layer_data != q_layer_data.round()).any():
                raise Exception("Quantized weights of {} layer include non-integer values".format(layer.__class__.__name__))

def initial_input_scale(pixels: np.ndarray) -> float:
    '''
    Calculate a scaling factor for the images that are input to the first layer of the CNN.

    Parameters:
    pixels (ndarray): The values of all the pixels which were part of the input image during training

    Returns:
    float: A scaling factor that the input should be multiplied by before being fed into the first layer.
            This value does not need to be an 8-bit integer.
    '''
    
    max_ = pixels.max().item()
    min_ = pixels.min().item()

    max_mag = max(abs(max_), abs(min_))
    range = max_mag
    scale = 127.0 / range
    scale = closest_lower_power_of_2(scale)

    return scale

def activation_scale(activations: np.ndarray, n_w: float, n_initial_input: float, ns: List[Tuple[float, float]]) -> float:
    '''
    Calculate a scaling factor to multiply the output of a layer by.

    Parameters:
    activations (ndarray): The values of all the pixels which have been output by this layer during training
    n_w (float): The scale by which the weights of this layer were multiplied as part of the "quantize_weights" function you wrote earlier
    n_initial_input (float): The scale by which the initial input to the neural network was multiplied
    ns ([(float, float)]): A list of tuples, where each tuple represents the "weight scale" and "output scale" (in that order) for every preceding layer

    Returns:
    float: A scaling factor that the layer output should be multiplied by before being fed into the first layer.
            This value does not need to be an 8-bit integer.
    '''

    activations *= n_initial_input

    for nw, nout in ns:
        activations *= nw * nout

    activations *= n_w

    max_ = activations.max().item()
    min_ = activations.min().item()

    max_mag = max(abs(max_), abs(min_))
    range = max_mag
    
    scale = 127.0 / range
    scale = closest_lower_power_of_2(scale)

    return scale

def quantize_activations(model: nn.Module):
    quantized_layers = [child for child in flattened_children(model) if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear)]

    # Quantize the initial inputs
    input_layer = quantized_layers[0]
    input_activations = model.activations[0]
    model.input_scale = initial_input_scale(input_activations)

    def initial_input_scaling_hook(layer: nn.Module, x):
        x = x[0]

        result = torch.clamp(x * model.input_scale, min=-128, max=127).round()

        # Add this code just in case
        if (result < -128).any() or (result > 127).any():
            raise Exception("Input to the input {} layer is out of bounds for an 8-bit signed integer".format(l.__class__.__name__))
        if (result != result.round()).any():
            raise Exception("Input to {} layer has non-integer values".format(l.__class__.__name__))

        return result

    input_layer.register_forward_pre_hook(initial_input_scaling_hook)

    # Now, quantize all the outputs
    output_activations = model.activations[1:]

    def make_output_scaling_hook(layer: nn.Module):
        def output_scaling_hook(layer: nn.Module, x, y):
            result = torch.clamp(y * layer.output_scale, min=-128, max=127).round()
            # layer.saved = [x[0].clone(), y.clone(), result.clone()]
            x = x[0]
            # import pdb; pdb.set_trace()
            if (x < -128).any() or (x > 127).any():
                # import pdb; pdb.set_trace()
                raise Exception("Input to {} layer is out of bounds for an 8-bit signed integer".format(layer.__class__.__name__))
            if (x != x.round()).any():
                raise Exception("Input to {} layer has non-integer values".format(layer.__class__.__name__))

            return result
        return output_scaling_hook

    preceding_layer_scales = []
    for layer, act in zip(quantized_layers, output_activations):
        layer.output_scale = activation_scale(act, layer.weight.scale, model.input_scale, preceding_layer_scales)
        layer.register_forward_hook(make_output_scaling_hook(layer))
        preceding_layer_scales.append((layer.weight.scale, layer.output_scale))
    
    # bn_relu_layers = [child for child in flattened_children(model) if isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.ReLU)]
    # # This is added for due to potential numerical instability
    # def output_scaling_hook_bn_relu(layer: nn.Module, x, y):
    #     result = torch.clamp(y, min=-128, max=127).round()
    #     # layer.saved = [x[0].clone(), y.clone(), result.clone()]
    #     return result
    # for layer in bn_relu_layers:
    #     layer.register_forward_hook(output_scaling_hook_bn_relu)


def register_activation_profiling_hooks(model: nn.Module):
    model.activations = []

    def profile_hook(model: nn.Module, idx: int, final_idx: int = None):
        def hook(m, x, y):
            model.activations[idx] = np.append(model.activations[idx], x[0].detach().cpu().reshape(-1))
            if final_idx is not None:
                model.activations[final_idx] = np.append(model.activations[final_idx], y[0].detach().cpu().reshape(-1))
                
            m.input_shape = tuple(x[0].shape)
        return hook

    layers = flattened_children(model)
    layers = [layer for layer in layers if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]

    handles = []

    for i, layer in enumerate(layers):
        model.activations.append(np.empty(0))
        final_idx = None

        if i+1 == len(layers):
            model.activations.append(np.empty(0))
            final_idx = i+1
        
        handle = layer.register_forward_hook(profile_hook(model, i, final_idx))
        handles.append(handle)

    def remove_all_handles():
        for handle in handles:
            handle.remove()
    
    return remove_all_handles

def quantized_bias(bias: torch.Tensor, n_w: float, n_initial_input: float, ns: List[Tuple[float, float]]) -> torch.Tensor:
    '''
    Quantize the weights so that all values are integers between -128 and 127.

    Parameters:
    bias (Tensor): The floating point values of the bias
    n_w (float): The scale by which the weights of this layer were multiplied
    n_initial_input (float): The scale by which the initial input to the neural network was multiplied
    ns ([(float, float)]): A list of tuples, where each tuple represents the "weight scale" and "output scale" (in that order) for every preceding layer

    Returns:
    Tensor: The bias in quantized form, where every value is an integer between -128 and 127.
            The "dtype" will still be "float", but the values themselves should all be integers.
    '''

    scale = n_initial_input

    for nw, nout in ns:
        scale *= nw * nout

    scale *= n_w

    return torch.clamp(bias * scale, min=-2147483648, max=2147483647).round(), scale

def quantize_layer_biases(model: nn.Module):
    preceding_layer_scales = []

    for layer in flattened_children(model):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # print(layer)

            if layer.bias is not None:
                q_layer_data, scale = quantized_bias(layer.bias.data, layer.weight.scale, model.input_scale, preceding_layer_scales)
                q_layer_data = q_layer_data.to(device)

                layer.bias.data = q_layer_data
                layer.bias.scale = scale

                if (q_layer_data < -2147483648).any() or (q_layer_data > 2147483647).any():
                    raise Exception("Quantized bias of {} layer include values out of bounds for a 32-bit signed integer".format(layer.__class__.__name__))
                if (q_layer_data != q_layer_data.round()).any():
                    raise Exception("Quantized bias of {} layer include non-integer values".format(layer.__class__.__name__))

            preceding_layer_scales.append((layer.weight.scale, layer.output_scale))

def quantize_bottleneck_layers(model: nn.Module):
    for child in model.modules():
        if isinstance(child, torchvision.models.resnet.Bottleneck):
            if hasattr(child, "quantized") and child.quantized:
                raise Exception("bottleneck ALREADY quantized")
            child.quantized = True

            child.register_forward_hook(lambda layer, x, y: torch.clamp(y, min=-128, max=127))
        elif isinstance(child, torchvision.models.resnet.BasicBlock):
            raise Exception("basic blocks not yet quantized")

def quantize_averaging_layer(model: nn.Module):
    # Quantize outputs of averaging layer
    model.model.fc.register_forward_pre_hook(lambda layer, x: x[0].round())

def test(model, dataloader, max_iter=40):
    for idx, (images, _, _) in enumerate(dataloader):
        images = images.to(device)
        images_nchw = (images.type(torch.float32)/255).permute(0,3,1,2)
        images_nchw.requires_grad = False
        _ = model(images_nchw)
        if idx >= max_iter:
            break

if __name__ == "__main__":
    model = get_model()
    vsa = get_vsa(device)
    train_dl = get_test_data(vsa)
    model.eval()
    sample_input = torch.ones(1,3,25,25, device=device)
    print(model(sample_input))
    merge_biases(model)
    remove_biases(model)
    print(model(sample_input))
    stop_profiling = register_activation_profiling_hooks(model)
    test(model, train_dl, max_iter=10)
    print("Stop profiling")
    stop_profiling()
    print("Quantize weights")
    quantize_layer_weights(model)
    print("Quantize activations")
    quantize_activations(model)
    print("Quantize bottleneck layers")
    quantize_bottleneck_layers(model)
    print("Quantize averaging layer")
    quantize_averaging_layer(model)
    print("Quantize biases")
    quantize_layer_biases(model)
    print(model(sample_input))
    # import pdb; pdb.set_trace()

    # print(model)