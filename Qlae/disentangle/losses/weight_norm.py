import torch

def get_model_layers_with_name(model, name):
    has_name = lambda x: hasattr(x, name)
    layers = [layer for layer in model.children() if has_name(layer)]
    return layers

def mean_squared_weight_norm(model):
    sum_of_square = 0
    num = 0
    for name in ['weight', 'bias']:
        layers = get_model_layers_with_name(model, name)
        for layer in layers:
            array = getattr(layer, name)
            if isinstance(array, torch.Tensor):
                sum_of_square += torch.sum(array**2).item()
                num += array.numel()
    return sum_of_square / num

def mean_absolute_weight_norm(model):
    sum_of_abs = 0
    num = 0
    for name in ['weight', 'bias']:
        layers = get_model_layers_with_name(model, name)
        for layer in layers:
            array = getattr(layer, name)
            if isinstance(array, torch.Tensor):
                sum_of_abs += torch.sum(torch.abs(array)).item()
                num += array.numel()
    return sum_of_abs / num
