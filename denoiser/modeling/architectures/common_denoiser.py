import torch.nn as nn
from ..networks import build_base_module
from ..denoiser_head import build_denoiser_head


class CommonDenoiser(nn.Module):

    def __init__(self, base_net, denoiser_head, **kwargs):
        super(CommonDenoiser, self).__init__()
        self.base_net = build_base_module(base_net)
        self.denoiser_head = build_denoiser_head(denoiser_head)

    def forward_train(self, base_output, target=None, mask=None):

        loss = self.denoiser_head.loss(base_output, target, mask)

        return loss

    def forward_test(self, base_output):
        out = self.denoiser_head(base_output)
        return out

    def forward(self, input_data, target=None, mask=None, **kwargs):

        base_output = self.base_net(input_data)
        if self.training:
            return self.forward_train(base_output, target, mask)
        else:
            return self.forward_test(base_output)
