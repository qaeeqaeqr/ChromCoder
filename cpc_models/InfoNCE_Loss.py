import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


class InfoNCE_Loss(nn.Module):
    """Performs predictions and InfoNCE Loss
    Args:
        pred_steps (int): number of steps into the future to perform predictions
        neg_samples (int): number of negative samples to be used for contrastive loss
        in_channels (int): number of channels of input tensors (size of encoding vector from encoder network and autoregressive network)
    """

    def __init__(self, pred_steps, neg_samples, in_channels):
        super().__init__()

        self.pred_steps = pred_steps
        self.neg_samples = neg_samples

        self.W_k = nn.ModuleList(
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
            for _ in range(self.pred_steps)
        )

        self.contrast_loss = ExpNLLLoss()

    def forward(self, z, c, skip_step=1):
        """
        shape of z = shape of c = (batch_size, encoding_size, grid, grid)
        :param z:
        :param c:
        :param skip_step:
        :return:
        """
        batch_size = z.shape[0]
        total_loss = 0
        cur_device = z.get_device()

        # For each element in c, contrast with elements below
        for k in range(1, self.pred_steps + 1):
            ### compute log f(c_t, x_{t+k}) = z^T_{t+k} * W_k * c_t

            ### compute z^T_{t+k} * W_k:
            ztwk = (
                self.W_k[k - 1]
                    .forward(
                    z[:, :, (k + skip_step):, :])  # for patched chromosome, only the first column should be focused on
                    .permute(2, 3, 0, 1)  # H, W, Bx, C
                    .contiguous()
            )  # y, x, b, c
            # print('ztwk.shape:', ztwk.shape)

            ztwk_shuf = ztwk.view(
                ztwk.shape[0] * ztwk.shape[1] * ztwk.shape[2], ztwk.shape[3]
            )  # y * x * batch, c
            # print('ztwk_shuf.shape:', ztwk_shuf.shape)
            rand_index = torch.randint(
                ztwk_shuf.shape[0],  # y *  x * batch
                (ztwk_shuf.shape[0] * self.neg_samples, 1),
                dtype=torch.long,
                device=cur_device,
            )
            # print('rand_index.shape:', rand_index.shape)
            # Sample more
            rand_index = rand_index.repeat(1, ztwk_shuf.shape[1])

            ztwk_shuf = torch.gather(
                ztwk_shuf, dim=0, index=rand_index, out=None
            )  # y * x * b * n, c

            ztwk_shuf = ztwk_shuf.view(
                ztwk.shape[0],
                ztwk.shape[1],
                ztwk.shape[2],
                self.neg_samples,
                ztwk.shape[3],
            ).permute(
                0, 1, 2, 4, 3
            )  # y, x, b, c, n

            ### Compute  x_W1 * c_t:
            context = (
                c[:, :, :-(k + skip_step), :].permute(2, 3, 0, 1).unsqueeze(-2)
            )  # y, x, b, 1, c
            # print('context.shape:', context.shape)
            log_fk_main = torch.matmul(context, ztwk.unsqueeze(-1)).squeeze(
                -2
            )  # y, x, b, 1

            log_fk_shuf = torch.matmul(context, ztwk_shuf).squeeze(-2)  # y, x, b, n

            log_fk = torch.cat((log_fk_main, log_fk_shuf), 3)  # y, x, b, 1+n
            log_fk = log_fk.permute(2, 3, 0, 1)  # b, 1+n, y, x

            log_fk = torch.softmax(log_fk, dim=1)

            true_f = torch.zeros(
                (batch_size, log_fk.shape[-2], log_fk.shape[-1]),
                dtype=torch.long,
                device=cur_device,
            )  # b, y, x

            total_loss += self.contrast_loss(input=log_fk, target=true_f)

        total_loss /= self.pred_steps

        return total_loss


class ExpNLLLoss(_WeightedLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(ExpNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        x = torch.log(input + 1e-11)
        return F.nll_loss(x, target, weight=self.weight, ignore_index=self.ignore_index,
                          reduction=self.reduction)


class PositionalPredictiveLoss(nn.Module):

    def __init__(self, pred_steps, neg_samples, in_channels,
                 positional=0.1, predictive=1,
                 grid_size=13):
        """

        Args:
            positional: ratio of positional loss
            predictive: ratio of predictive loss
        """
        super().__init__()

        self.pred_steps = pred_steps
        self.neg_samples = neg_samples

        self.W_k = nn.ModuleList(
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
            for _ in range(self.pred_steps)
        )

        self.predictive_loss = ExpNLLLoss()
        self.positional_loss = nn.CrossEntropyLoss()

        self.positional = positional
        self.predictive = predictive
        self.grid_size = grid_size
        self.pos_pred = pred_steps

        self.linear = nn.Linear(self.pos_pred * in_channels, self.pred_steps)

    def forward(self, z, c, skip_step=1):
        """
        shape of z = shape of c = (batch_size, encoding_size, grid, grid)
        :param z:
        :param c:
        :param skip_step:
        :return:
        """
        batch_size = z.shape[0]
        total_loss = 0
        cur_device = z.get_device()

        # For each element in c, contrast with elements below
        for k in range(1, self.pred_steps + 1):
            ### compute log f(c_t, x_{t+k}) = z^T_{t+k} * W_k * c_t

            ### compute z^T_{t+k} * W_k:
            ztwk = (
                self.W_k[k - 1]
                    .forward(
                    z[:, :, (k + skip_step):, :])  # for patched chromosome, only the first column should be focused on
                    .permute(2, 3, 0, 1)  # H, W, Bx, C
                    .contiguous()
            )  # y, x, b, c
            # print('ztwk.shape:', ztwk.shape)

            ztwk_shuf = ztwk.view(
                ztwk.shape[0] * ztwk.shape[1] * ztwk.shape[2], ztwk.shape[3]
            )  # y * x * batch, c
            # print('ztwk_shuf.shape:', ztwk_shuf.shape)
            rand_index = torch.randint(
                ztwk_shuf.shape[0],  # y *  x * batch
                (ztwk_shuf.shape[0] * self.neg_samples, 1),
                dtype=torch.long,
                device=cur_device,
            )
            # print('rand_index.shape:', rand_index.shape)
            # Sample more
            rand_index = rand_index.repeat(1, ztwk_shuf.shape[1])

            ztwk_shuf = torch.gather(
                ztwk_shuf, dim=0, index=rand_index, out=None
            )  # y * x * b * n, c

            ztwk_shuf = ztwk_shuf.view(
                ztwk.shape[0],
                ztwk.shape[1],
                ztwk.shape[2],
                self.neg_samples,
                ztwk.shape[3],
            ).permute(
                0, 1, 2, 4, 3
            )  # y, x, b, c, n

            ### Compute  x_W1 * c_t:
            context = (
                c[:, :, :-(k + skip_step), :].permute(2, 3, 0, 1).unsqueeze(-2)
            )  # y, x, b, 1, c
            # print('context.shape:', context.shape)
            log_fk_main = torch.matmul(context, ztwk.unsqueeze(-1)).squeeze(
                -2
            )  # y, x, b, 1

            log_fk_shuf = torch.matmul(context, ztwk_shuf).squeeze(-2)  # y, x, b, n

            log_fk = torch.cat((log_fk_main, log_fk_shuf), 3)  # y, x, b, 1+n
            log_fk = log_fk.permute(2, 3, 0, 1)  # b, 1+n, y, x

            log_fk = torch.softmax(log_fk, dim=1)

            true_f = torch.zeros(
                (batch_size, log_fk.shape[-2], log_fk.shape[-1]),
                dtype=torch.long,
                device=cur_device,
            )  # b, y, x

            total_loss += self.predictive * self.predictive_loss(input=log_fk,
                                                                 target=true_f)

            # note:   computing predictive loss above
            # --------------------------------------------------------------------------------- #
            # note:   computing positional loss below

            context = c[:, :, -(k + skip_step + self.pos_pred):-(k + skip_step), :].squeeze(-1)
            context = context.reshape(context.shape[0], context.shape[1] * context.shape[2])
            # print(context.shape)

            pred_position = self.linear.forward(context)

            true_position = torch.full(size=[int(context.shape[0]), ],
                                       fill_value=k - 1,
                                       dtype=torch.long,
                                       device=cur_device)

            total_loss += self.positional * self.positional_loss(input=pred_position,
                                                                 target=true_position)

        total_loss /= self.pred_steps

        return total_loss
