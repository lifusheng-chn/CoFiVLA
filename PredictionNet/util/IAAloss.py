import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

eps = 1e-8


class IAALoss(torch.nn.Module):
    def __init__(self, loss_type='js'):
        super(IAALoss, self).__init__()
        self.loss_type = loss_type
        self.celoss = nn.CrossEntropyLoss()
        # self.logceloss = LogitNormLoss()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y):
        # loss=reshape
        outputs_without_aux = {k: v for k, v in y_pred.items() if k != 'aux_outputs'}
        y = y.view(-1, 10)  # ori
        # y = y.view(-1, 1)

        # out_prob, aadb_prob = outputs_without_aux['pred_logits']
        out_prob = outputs_without_aux['pred_logits']

        loss = self.loss_func(out_prob, y)
        # loss = self.loss_func(out_prob, y)
        if 'aux_outputs' in y_pred:
            for aux_outputs in y_pred['aux_outputs']:
                loss += self.loss_func(aux_outputs['pred_logits'], y)
        return loss

    def get_target(self, outputs, targets):
        src_logits = outputs['pred_logits']
        target_classes = torch.cat([torch.full((1, src_logits.shape[1]), target,
                                               dtype=torch.float32, device=src_logits.device) for target in targets])
        return target_classes

    def loss_func(self, y_pred, y):
        if self.loss_type == 'mae':
            loss = F.l1_loss(y_pred, y)
        elif self.loss_type == 'mse':
            loss = F.mse_loss(y_pred, y)
        elif self.loss_type == 'js':
            loss = js_loss(y, y_pred)
        elif self.loss_type == 'js_bce':
            loss1 = js_loss(y, y_pred)
            loss2 = self.bceloss(y, y_pred)
            loss = 0.9 * loss1 + 0.1 * loss2
        elif self.loss_type == 'js_mse':
            device = y_pred.device
            label_mos = (y * torch.arange(1, 11).to(device)).sum(dim=1)
            out_mos = (y_pred * torch.arange(1, 11).to(device)).sum(dim=1)
            loss1 = js_loss(y, y_pred)
            loss2 = self.mse(label_mos, out_mos)
            loss = 0.9 * loss1 + 0.1 * loss2
        elif self.loss_type == 'ce':
            loss = self.celoss(y_pred, y)

        elif self.loss_type == 'logce':
            device = y_pred.device
            label_mos = (y * torch.arange(1, 11).to(device)).sum(dim=1)
            out_mos = (y_pred * torch.arange(1, 11).to(device)).sum(dim=1)
            # loss1 = self.mse(label_mos, out_mos)
            loss2 = self.logceloss(y_pred, y.argmax(dim=-1))
            loss = loss2

        elif self.loss_type == 'js_aadb':
            device = y_pred.device
            label_mos = (y * torch.arange(1, 11).to(device)).sum(dim=1)
            loss1 = F.mse_loss(label_mos, y_pred)
            loss2 = F.mse_loss(aadb_prob, aadb_label)
            # print("loss1:",loss1)
            # print("loss2:",loss2)
            loss = loss1 + loss2

        elif self.loss_type == 'js_content':
            loss1 = js_loss(y, y_pred)
            loss2 = F.cross_entropy(aadb_prob, aadb_label.squeeze().long())
            loss = loss1 + 0.01 * loss2

        return loss

    # def loss_func(self, y_label, y1, y2, y3):  # mmca
    #     if self.loss_type == 'img_text_fuse_loss':
    #         loss1 = js_loss(y1, y_label)
    #         loss2 = js_loss(y2, y_label)
    #         loss3 = js_loss(y3, y_label)
    #         loss = 0.8 * loss1 + 0.1 * loss2 + 0.1 * loss3
    #
    #     return loss


class LogitNormLoss(nn.Module):

    def __init__(self, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)


def js_loss(p, q):
    # p.shape == q.shape == (bs, dim)
    mask_p = (p > 1e-5).float()
    mask_q = (q > 1e-5).float()
    log_2 = torch.log(torch.tensor(2.).to(q.device))
    _p = p * mask_p + (1 - mask_p)
    _q = q * mask_q + (1 - mask_q)
    loss1 = p * torch.log(2 * _p) + q * torch.log(2 * _q) - (p + q) * torch.log(_p + _q)
    loss2 = loss1 * mask_p * mask_q
    loss3 = loss2 + (1 - mask_p) * (q * log_2)
    loss4 = loss3 + (1 - mask_q) * (p * log_2)
    loss5 = torch.sum(loss4, dim=1)
    loss6 = .5 * torch.mean(loss5)
    return loss6

class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]

        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]

        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=1))
        return samplewise_emd.mean()

class EMDLoss1(nn.Module):
    def __init__(self):
        super(EMDLoss1, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]

        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]

        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.mean(torch.pow(torch.abs(cdf_diff), 1), dim=1)
        return samplewise_emd.mean()


if __name__ == '__main__':
    # y = torch.tensor([[1.2, 2.3, 3.4]])
    y_pred = torch.tensor([[1.3, 2.4, 3.5], [1.3, 3.5, 2.4]])
    y = torch.tensor([2, 2])
    logce = LogitNormLoss()
    loss = logce(y_pred, y)
    print(loss)
