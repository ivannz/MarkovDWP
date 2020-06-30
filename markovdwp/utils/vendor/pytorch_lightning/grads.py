"""
Module to describe gradients
"""
from typing import Dict

import torch


class GradInformation(torch.nn.Module):

    def grad_norm(self, norm_type: float) -> Dict[str, int]:
        norms, all_norms = {}, []
        for name, p in self.named_parameters():
            if p.grad is None:
                continue

            param_norm = float(p.grad.data.norm(norm_type))
            all_norms.append(param_norm)

            norms[f'grad_{norm_type}_norm/{name}'] = param_norm

        total_norm = float(torch.tensor(all_norms).norm(norm_type))
        norms[f'grad_{norm_type}_norm_total'] = total_norm

        return norms
