import torch.nn as nn


class DenoiserHead(nn.Module):
    def __init__(self, head_type, loss_type, loss_weight):

        super(DenoiserHead, self).__init__()
        self.head_type = head_type
        self.loss_type = loss_type
        if loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction='mean')
        elif loss_type == "l1":
            self.loss_func = nn.L1Loss()
        else:
            raise TypeError

        self.loss_wight = loss_weight

    def forward(self, base_output):
        if self.head_type == "supervise":
            output = base_output
        else:
            raise TypeError

        return output

    def loss(self, base_output, target=None, mask=None):
        output = self.forward(base_output)
        loss = {}
        model_info = {}

        if mask is not None:
            output = output*mask
            target = target*mask

        loss[self.loss_type] = self.loss_func(output, target) * self.loss_wight[self.loss_type]
        loss["loss_total"] = loss[self.loss_type]

        return loss, model_info
