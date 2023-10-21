import numpy as np
from models.resnet import Bottleneck
from models.nn_non_decomposed import MultiConceptNonDecomposed
import torch
import torchvision
from typing import List, Tuple
from torch import nn
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# def get_model():
#     model = MultiConceptNonDecomposed(dim=DIM, device=device)
#     return model

def get_cos_similarity(input, others):
    """
    Return the cosine similarity.
    """
    input_dot = torch.sum(input * input, dim=-1)
    input_mag = torch.sqrt(input_dot)
    others_dot = torch.sum(others * others, dim=-1)
    others_mag = torch.sqrt(others_dot)
    if input.dim() >= 2:
        magnitude = input_mag.unsqueeze(-1) * others_mag.unsqueeze(-2)
    else:
        magnitude = input_mag * others_mag
    magnitude = torch.clamp(magnitude, min=1e-08)

    if others.dim() >= 2:
        others = others.transpose(-2, -1)

    return torch.matmul(input.type(torch.float32), others.type(torch.float32)) / magnitude

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
        
    # return next(result)
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
    scale = 128.0 / range
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
    
    range = pixels.item()
    scale = 128.0 / range
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

    range = activation_scale.item()
    
    scale = 128.0 / range
    scale = closest_lower_power_of_2(scale)

    return scale

def quantize_activations(model: nn.Module):
    quantized_layers = [child for child in flattened_children(model) if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear)]

    # Quantize the initial inputs
    input_layer = quantized_layers[0]
    input_activations = model.activations[0]
    model.input_scale = initial_input_scale(input_activations)
    model.quantized_output = []

    def initial_input_scaling_hook(layer: nn.Module, x):
        x = x[0]
        result = x * model.input_scale
        result = torch.clamp(result, min=-128, max=127).round()

        # Add this code just in case
        if (result < -128).any() or (result > 127).any():
            raise Exception("Input to the input {} layer is out of bounds for an 8-bit signed integer".format(layer.__class__.__name__))
        if (result != result.round()).any():
            raise Exception("Input to {} layer has non-integer values".format(layer.__class__.__name__))

        return result

    input_layer.register_forward_pre_hook(initial_input_scaling_hook)

    # Now, quantize all the outputs
    output_activations = model.activations[1:]

    def make_output_scaling_hook(layer: nn.Module):
        def output_scaling_hook(layer: nn.Module, x, y):
            result = y * layer.output_scale
            result = torch.clamp(result, min=-128, max=127).round()
            model.quantized_output.append(result)

            x = x[0]
            if (x < -128).any() or (x > 127).any():
                # import pdb; pdb.set_trace()
                raise Exception("Input to {} layer is out of bounds for an 8-bit signed integer".format(layer.__class__.__name__))
            if (x != x.round()).any():
                raise Exception("Input to {} layer has non-integer values".format(layer.__class__.__name__))
            return result
        
        # def final_output_scaling_hook(layer: nn.Module, x, y):
        #     result = output_scaling_hook(layer, x, y)
        #     result = result / 128.0 * layer.output_scale
        #     return result

        return output_scaling_hook

    preceding_layer_scales = []
    for idx, (layer, act) in enumerate(zip(quantized_layers, output_activations)):
        layer.output_scale = activation_scale(act, layer.weight.scale, model.input_scale, preceding_layer_scales)
        layer.register_forward_hook(make_output_scaling_hook(layer))
        preceding_layer_scales.append((layer.weight.scale, layer.output_scale))

        total_scale = model.input_scale
        for i, (nw, nout) in enumerate(preceding_layer_scales):
            if i not in [4, 11, 18, 25]:
                total_scale *= nw * nout
        layer.total_scale = total_scale

    return total_scale

 
def register_activation_profiling_hooks(model: nn.Module):
    model.activations = []
    model.output_activations = []
    def profile_hook(model: nn.Module, idx: int, final_idx: int = None):
        def hook(m, x, y):
            model.activations[idx] = torch.max(model.activations[idx], x[0].detach().cpu().reshape(-1).abs().max())
            model.output_activations[idx].append(y.detach().cpu())

            if final_idx is not None:
                model.activations[final_idx] = torch.max(model.activations[final_idx], y[0].detach().cpu().reshape(-1).abs().max())

            m.input_shape = tuple(x[0].shape)
        return hook

    layers = flattened_children(model)
    layers = [layer for layer in layers if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]

    handles = []

    for i, layer in enumerate(layers):
        model.activations.append(torch.tensor(0))
        # model.activations.append(np.empty(0))
        model.output_activations.append([])

        final_idx = None

        if i+1 == len(layers):
            model.activations.append(torch.tensor(0))
            # model.activations.append(np.empty(0))
            model.output_activations.append([])
            final_idx = i+1
        
        handle = layer.register_forward_hook(profile_hook(model, i, final_idx))
        handles.append(handle)

    def remove_all_handles():
        for handle in handles:
            handle.remove()
    
    return remove_all_handles

# def quantized_bias(bias: torch.Tensor, n_w: float, n_initial_input: float, ns: List[Tuple[float, float]]) -> torch.Tensor:
#     '''
#     Quantize the bias.

#     Parameters:
#     bias (Tensor): The floating point values of the bias
#     n_w (float): The scale by which the weights of this layer were multiplied
#     n_initial_input (float): The scale by which the initial input to the neural network was multiplied
#     ns ([(float, float)]): A list of tuples, where each tuple represents the "weight scale" and "output scale" (in that order) for every preceding layer

#     Returns:
#     Tensor: The bias in quantized form, where every value is an integer between -128 and 127.
#             The "dtype" will still be "float", but the values themselves should all be integers.
#     '''
#     scale = n_initial_input

#     for nw, nout in ns:
#         scale *= nw * nout

#     scale *= n_w

#     return torch.clamp(bias * scale, min=-2147483648, max=2147483647).round(), scale

def quantized_bias(bias: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.clamp(bias * scale, min=-2147483648, max=2147483647).round()

def quantize_layer_biases(model: nn.Module):
    preceding_layer_scales = []

    for layer in flattened_children(model):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # print(layer)

            if layer.bias is not None:
                scale = layer.total_scale / layer.output_scale
                q_layer_data = quantized_bias(layer.bias.data, scale)
                # q_layer_data, scale = quantized_bias(layer.bias.data, layer.weight.scale, model.input_scale, preceding_layer_scales)
                q_layer_data = q_layer_data.to(device)

                layer.bias.data = q_layer_data
                layer.bias.scale = scale

                if (q_layer_data < -2147483648).any() or (q_layer_data > 2147483647).any():
                    raise Exception("Quantized bias of {} layer include values out of bounds for a 32-bit signed integer".format(layer.__class__.__name__))
                if (q_layer_data != q_layer_data.round()).any():
                    raise Exception("Quantized bias of {} layer include non-integer values".format(layer.__class__.__name__))

            preceding_layer_scales.append((layer.weight.scale, layer.output_scale))
    # for idx, layer in enumerate(quantized_layers):
    #     if layer.bias is not None:
    #         scale = layer.total_scale / layer.output_scale
    #         q_layer_data = quantized_bias(layer.bias.data, scale)
    #         q_layer_data = q_layer_data.to(device)

    #         layer.bias.data = q_layer_data
    #         layer.bias.scale = scale

    #         if (q_layer_data < -2147483648).any() or (q_layer_data > 2147483647).any():
    #             raise Exception("Quantized bias of {} layer include values out of bounds for a 32-bit signed integer".format(layer.__class__.__name__))
    #         if (q_layer_data != q_layer_data.round()).any():
    #             raise Exception("Quantized bias of {} layer include non-integer values".format(layer.__class__.__name__))

def quantize_bottleneck_layers(model: nn.Module):
    for child in model.modules():
        if isinstance(child, Bottleneck):
            if child.quantized:
                raise Exception("bottleneck ALREADY quantized")
            child.quantized = True

            child.register_forward_hook(lambda layer, x, y: torch.clamp(y, min=-128, max=127))
        elif isinstance(child, torchvision.models.resnet.BasicBlock):
            raise Exception("basic blocks not yet quantized")

def quantize_averaging_layer(model: nn.Module):
    # Quantize outputs of averaging layer
    model.model.fc.register_forward_pre_hook(lambda layer, x: x[0].round())


def final_output_scaling(output: torch.Tensor, range: int):
    assert (output == torch.clamp(output, min=-128, max=127).round()).all()

    result = output / 128.0 * range
    return result

def test(model, dataloader, max_iter=40, output_scale=None):
    loss_fn = torch.nn.MSELoss()
    losses = []
    sims = []
    for idx, (images, labels, targets, _) in enumerate(dataloader):
        with torch.no_grad():
            images = transforms.ConvertImageDtype(torch.float32)(images.to(device))
            targets_float = targets.to(device).type(torch.float32)
            output = model(images)
            # print([round(e, 3) for e in output.tolist()[0]])
            # if idx >= max_iter:
            #     print(output, output.max(), output.min())
            # if quantized:
            #     output = final_output_scaling(output, dataloader.dataset.max_num_objects)
            if output_scale is not None:
                output = output / output_scale
            loss = loss_fn(output, targets_float)
            losses.append(loss.item())
            sim = torch.sum(get_cos_similarity(output.round(), targets_float)).item()
            sims.append(sim)
            if idx >= max_iter:
                print(output)
                print(targets_float)
                break
    print(sum(sims)/len(sims))
    print(sum(losses)/len(losses))

def profile_model(model: nn.Module, dataloader):
    """
    Profiles the model to get the activation range and input shape of each layer.
    Also merges the batchnorm layers into the previous convolutional layer.
    """
    merge_biases(model)
    remove_biases(model)
    stop_profiling = register_activation_profiling_hooks(model)
    test(model, dataloader, max_iter=99)
    stop_profiling()

def quantize_model(model: nn.Module, dataloader):
    model.eval()

    profile_model(model, dataloader)

    print("Quantize weights")
    quantize_layer_weights(model)
    print("Quantize activations")
    total_scale = quantize_activations(model)
    print("Total scale: {}".format(total_scale))
    print("Quantize bottleneck layers")
    quantize_bottleneck_layers(model)
    print("Quantize averaging layer")
    quantize_averaging_layer(model)
    print("Quantize biases")
    quantize_layer_biases(model)
    test(model, dataloader, max_iter=99, output_scale=total_scale)
    # quantized_layers = [child for child in flattened_children(model) if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear)]
    # for i in range(len(quantized_layers)):
    #     print("*" * 10 + str(i) + "*" * 10)
    #     layer = quantized_layers[i]
    #     print(model.quantized_output[i][0].reshape(-1) / layer.total_scale)
    #     print(model.output_activations[i][0].reshape(-1))