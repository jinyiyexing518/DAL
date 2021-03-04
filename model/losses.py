import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.bbox import bbox_overlaps, min_area_square
from utils.box_coder import BoxCoder
from utils.overlaps.rbox_overlaps import rbox_overlaps
from utils.overlaps_cuda.rbbox_overlaps  import rbbx_overlaps


def xyxy2xywh_a(query_boxes): # 返回回归值
    out_boxes = query_boxes.copy()
    out_boxes[:, 0] = (query_boxes[:, 0] + query_boxes[:, 2]) * 0.5
    out_boxes[:, 1] = (query_boxes[:, 1] + query_boxes[:, 3]) * 0.5
    out_boxes[:, 2] = query_boxes[:, 2] - query_boxes[:, 0]
    out_boxes[:, 3] = query_boxes[:, 3] - query_boxes[:, 1]
    return out_boxes

# cuda_overlaps
class IntegratedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, func = 'smooth'):
        super(IntegratedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.box_coder = BoxCoder()
        if func == 'smooth': # 损失函数转折处变得更加平滑
            self.criteron = smooth_l1_loss
        elif func == 'mse':
            self.criteron = F.mse_loss
        elif func == 'balanced':
            self.criteron = balanced_l1_loss

    def forward(self, classifications, regressions, anchors, refined_achors, annotations, \
                md_thres=0.5, mining_param=(1, 0., -1), ref=False):
        
        das = True
        cls_losses = []
        reg_losses = []
        batch_size = classifications.shape[0]
        alpha, beta, var = mining_param
#         import ipdb;ipdb.set_trace()
        for j in range(batch_size): #迭代每一张输入的图片
            # 分类和回归的特征图
            classification = classifications[j, :, :] # classification的维度=batch-size,box—num,概率
            regression = regressions[j, :, :] # regression的维度=batch-size,box-num,回归坐标值,包含角度
            # 真实的标注数据
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1] #类别不为-1的框
            if bbox_annotation.shape[0] == 0: # 如果目标数量为0
                cls_losses.append(torch.tensor(0).float().cuda())
                reg_losses.append(torch.tensor(0).float().cuda())
                continue
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4) #修剪在规定的范围之内，框的數量×類別數(2)

            # 下面计算匹配度,sa空间对齐,和输入ROI有关
            sa = rbbx_overlaps(xyxy2xywh_a(anchors[j, :, :].cpu().numpy()),
                               xyxy2xywh_a(bbox_annotation[:, :-1].cpu().numpy()),)
            if not torch.is_tensor(sa):
                # import ipdb;ipdb.set_trace()
                sa = torch.from_numpy(sa).cuda()
            if var != -1:
                # 下面计算特征对齐fa,是關於GT和回归之间的
                fa = rbbx_overlaps(xyxy2xywh_a(refined_achors[j, :, :].cpu().numpy()),
                                   xyxy2xywh_a(bbox_annotation[:, :-1].cpu().numpy()),)
                if not torch.is_tensor(fa):
                    fa = torch.from_numpy(fa).cuda()
                # 匹配度计算---空间对齐-特征对齐-惩罚项
                if var == 0:
                    md = abs((alpha * sa + beta * fa))
                else:# 匹配度计算---空间对齐-特征对齐-惩罚项
                    md = abs((alpha * sa + beta * fa) - abs(fa - sa)**var)
            else:
                das = False
                md = sa

            # 然后将所有目标压缩，我们不关注目标，只关注我们的anchor box和位置的目标之间的iou，然后就取最大值，所以维度是anchor box
            iou_max, iou_argmax = torch.max(md, dim=1) #应该是对于每一个gt中的目标，所有的anchor box都与其进行匹配，所以是anchor数量×目标数
            # 但是这里我们不关注每一个具体的目标类别-匹配度更加关注于是否阳性阴性样本-也就是重合度如何

            # 通过匹配性阈值来选取阳性样本，正样本-True的位置
            positive_indices = torch.ge(iou_max, md_thres)

            # 对于所有的anchor box计算匹配度后，max-gt返回的是对应目标数量的匹配度
            # argmax-gt返回的是相应的位置，也就是哪个预选框（其实下面这行代码就是选出和每一个gt目标匹配度最高的预选框）
            max_gt, argmax_gt = md.max(0)
            # import ipdb;ipdb.set_trace(context = 15)
            if (max_gt < md_thres).any(): # 都不及匹配度阈值
                positive_indices[argmax_gt[max_gt < md_thres]]=1 # 正样本中，超过阈值，最大阈值 = 1
              
            # matching-weight
            if das:
                pos = md[positive_indices]
                pos_mask = torch.ge(pos, md_thres)
                max_pos, armmax_pos = pos.max(0) # 这里就取得了阳性样本中的最大匹配值，以及其索引（位置）,用于后面计算补偿因子
                nt = md.shape[1] #gt中的目标数量
                for gt_idx in range(nt):
                    pos_mask[armmax_pos[gt_idx], gt_idx] = 1 # 这里的pos_mask对应维度 框数量×目标数量,也就是选出了每一个目标最大匹配度所对应的框，也就是一行一个True
                comp = torch.where(pos_mask, (1 - max_pos).repeat(len(pos),1), pos)
                # 对于拥有最大的匹配度的阳性样本-其计算损失的补偿因子(权重w)=1
                matching_weight = comp + pos # 然后再对其他的阳性样本进行补偿
            # import ipdb; ipdb.set_trace(context = 15)

            # cls loss
            cls_targets = (torch.ones(classification.shape) * -1).cuda()
            # 逐元素比较，小于md-thres -0.1就置零
            cls_targets[torch.lt(iou_max, md_thres - 0.1), :] = 0

            num_positive_anchors = positive_indices.sum() #阳性样本数量
            assigned_annotations = bbox_annotation[iou_argmax, :]
            cls_targets[positive_indices, :] = 0
            cls_targets[positive_indices, assigned_annotations[positive_indices, -1].long()] = 1

            alpha_factor = torch.ones(cls_targets.shape).cuda() * self.alpha

            # torch.where 类似于条件运算符，符合的保留第一个，否则第二个
            alpha_factor = torch.where(torch.eq(cls_targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(cls_targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
            # 交叉熵损失函数
            bin_cross_entropy = -(cls_targets * torch.log(classification+1e-6) + (1.0 - cls_targets) * torch.log(1.0 - classification+1e-6))
            if das:
                soft_weight = (torch.zeros(classification.shape)).cuda()
                soft_weight = torch.where(torch.eq(cls_targets, 0.), torch.ones_like(cls_targets), soft_weight)
                soft_weight[positive_indices, assigned_annotations[positive_indices, -1].long()] = (matching_weight.max(1)[0] + 1)
                # focal-loss加一个权重，该权重关注阳性样本
                cls_loss = focal_weight * bin_cross_entropy * soft_weight
            else:
                cls_loss = focal_weight * bin_cross_entropy
            # 这里计算不等于-1处的损失
            cls_loss = torch.where(torch.ne(cls_targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            # 好像这里的分类损失仅仅计算了补偿因子加权的损失
            cls_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # reg loss---回归的损失函数使用smoothL1损失-在转折处更加平滑
            if positive_indices.sum() > 0:
                all_rois = anchors[j, positive_indices, :] #阳性样本
                gt_boxes = assigned_annotations[positive_indices, :] #我们仅仅对阳性样本计算补偿加权的回归损失，所以这里按照positive-indices
                # reg-targets返回targets_dx, targets_dy, targets_dw, targets_dh, targets_dt用于计算回归损失
                reg_targets = self.box_coder.encode(all_rois, gt_boxes)
                if das: #加入补偿因子的损失计算, reg_target是偏移量，用于计算损失
                    reg_loss = self.criteron(regression[positive_indices, :], reg_targets, weight = matching_weight)
                else:
                    reg_loss = self.criteron(regression[positive_indices, :], reg_targets)
                reg_losses.append(reg_loss)

                if not torch.isfinite(reg_loss) :
                    import ipdb; ipdb.set_trace()
                k=1
            else:
                reg_losses.append(torch.tensor(0).float().cuda())

        loss_cls = torch.stack(cls_losses).mean(dim=0, keepdim=True) # 拼接后求均值
        loss_reg = torch.stack(reg_losses).mean(dim=0, keepdim=True) # 拼接后求均值
        return loss_cls, loss_reg

    
def smooth_l1_loss(inputs,
                   targets,
                   beta=1. / 9,
                   size_average=True,
                   weight = None):
    """
    https://github.com/facebookresearch/maskrcnn-benchmark
    """
    diff = torch.abs(inputs - targets)
    if  weight is  None:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
    else:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        ) * weight.max(1)[0].unsqueeze(1).repeat(1,5)
    if size_average:
        return loss.mean()
    return loss.sum()


def balanced_l1_loss(inputs,
                     targets,
                     beta=1. / 9,
                     alpha=0.5,
                     gamma=1.5,
                     size_average=True):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """
    assert beta > 0
    assert inputs.size() == targets.size() and targets.numel() > 0

    diff = torch.abs(inputs - targets)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    if size_average:
        return loss.mean()
    return loss.sum()

