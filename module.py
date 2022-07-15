# This file define custom functions with forward and backward passes.
# See https://pytorch.org/docs/stable/notes/extending.html for more details.
# Also https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
import torch
import contextlib
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model.inference(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model.inference(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model.inference(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
class L2ProjFunction(torch.autograd.Function):
    """
    This function defines the L2 projection for the input z.
    The forward pass uses a binary search and saves some quantities for the backward pass.
    The backward pass computes the Jacobian-vector product as in the Appendix of the paper.
    Note: if the forward pass loops forever, you may relax the termination condition a little bit.
    """

    @staticmethod
    def forward(self, z, dim=-1):

        z = z.transpose(dim, -1)
        left = torch.min(z, dim=-1, keepdim=True)[0]
        right = torch.max(z, dim=-1, keepdim=True)[0] + 1.0
        alpha_norm = torch.tensor(100.0, dtype=torch.float, device=z.device)
        one = torch.tensor(1.0, dtype=torch.float, device=z.device)
        # zero = torch.tensor(0.0, dtype=torch.float, device=z.device)
        # while not torch.allclose(right - left, zero):
        while not torch.allclose(alpha_norm, one):
            mid = left + (right - left) * 0.5
            alpha = torch.relu(mid - z)
            alpha_norm = torch.norm(alpha, dim=-1, keepdim=True)
            right[alpha_norm > 1.0] = mid[alpha_norm > 1.0]
            left[alpha_norm <= 1.0] = mid[alpha_norm <= 1.0]
        K = alpha.sum(-1, keepdim=True)
        alpha = alpha / K
        s = (alpha > 0).float()  # support, positivity mask
        zs = z * s
        S = s.sum(-1, keepdim=True)
        A = zs.sum(-1, keepdim=True) ** 2 - S * ((zs ** 2).sum(-1, keepdim=True) - 1)  # should have A > 0
        self.save_for_backward(alpha, K, s, S, A, torch.tensor(dim))
        return alpha.transpose(dim, -1)

    @staticmethod
    def backward(self, grad_output):

        alpha, K, s, S, A, dim = self.saved_tensors
        dim = dim.item()
        grad_output = grad_output.transpose(dim, -1)
        # first part
        vhat = (s * grad_output).sum(-1, keepdim=True) / S
        grad1 = (s / K) * (vhat - grad_output)
        # second part
        alpha_s = alpha * s - s / S
        grad2 = S / A.sqrt() * alpha_s * (alpha_s * grad_output).sum(-1, keepdim=True)

        return (grad1 - grad2).transpose(dim, -1)


class GradientReversalLayer(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(self, inputs):

        return inputs

    @staticmethod
    def backward(self, grad_output):

        grad_input = -grad_output.clone()
        return grad_input

