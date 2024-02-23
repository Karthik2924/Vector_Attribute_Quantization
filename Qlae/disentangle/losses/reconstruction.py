import torch
import torch.nn.functional as F

def binary_cross_entropy_loss(x_hat_logits, x_true_probs):
    return torch.mean(F.binary_cross_entropy_with_logits(x_hat_logits, x_true_probs))
