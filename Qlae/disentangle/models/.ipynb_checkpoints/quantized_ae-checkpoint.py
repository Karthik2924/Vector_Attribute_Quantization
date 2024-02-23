import torch
import torch.nn as nn
import torch.nn.functional as F
from . import ae
class QuantizedAE(ae.AE):
    @staticmethod
    def quantization_loss(continuous, quantized):
        return torch.mean(torch.square(continuous.detach() - quantized))

    @staticmethod
    def commitment_loss(continuous, quantized):
        return torch.mean(torch.square(continuous - quantized.detach()))

    def batched_loss(self, model, data, step, *args, key=None, **kwargs):
        outs = model(data['x'])
        quantization_loss = self.quantization_loss(outs['z_continuous'], outs['z_quantized'])
        commitment_loss = self.commitment_loss(outs['z_continuous'], outs['z_quantized'])
        binary_cross_entropy_loss = F.binary_cross_entropy_with_logits(outs['x_hat_logits'], data['x'].float(), reduction='mean')
        partition_norms = self.partition_norms()

        loss = (
            self.lambdas['binary_cross_entropy'] * binary_cross_entropy_loss +
            self.lambdas['quantization'] * quantization_loss +
            self.lambdas['commitment'] * commitment_loss
        )

        metrics = {
            'loss': loss.item(),
            'binary_cross_entropy_loss': binary_cross_entropy_loss.item(),
            'quantization_loss': quantization_loss.item(),
            'commitment_loss': commitment_loss.item(),
        }
        metrics.update({
            f'params_{k_partition}/{k_norm}': v.item()
            for k_norm, v_norm in partition_norms.items() for k_partition, v in v_norm.items()
        })

        aux = {
            'metrics': metrics,
            'outs': outs,
        }

        return loss, aux
