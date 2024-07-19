import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def F1_IoU_BCELoss(pred_masks, first_gt_mask):
    """
    binary cross entropy loss (iou loss) of the first frame for single sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks) # [bs*5, 1, 224, 224]
    indices = torch.tensor(list(range(0, len(pred_masks), 5)))
    indices = indices.cuda()

    first_pred = torch.index_select(pred_masks, dim=0, index=indices) # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1) # [bs, 1, 224, 224]

    
    # weight = torch.zeros_like(first_gt_mask).float().cuda()
    # weight = torch.fill_(weight,0.9)
    # weight[first_gt_mask>0]=1.1

    first_bce_loss = nn.BCELoss()(first_pred, first_gt_mask)

    return first_bce_loss


def DiceLoss(logits, targets):

    assert len(logits.shape) == 4
    indices = torch.tensor(list(range(0, len(logits), 5)))
    indices = indices.cuda()

    logits = torch.index_select(logits, dim=0, index=indices) # [bs, 1, 224, 224]
    assert logits.requires_grad == True, "Error when indexing predited masks"
    if len(targets.shape) == 5:
        targets = targets.squeeze(1) # [bs, 1, 224, 224]

    num = targets.size(0)
    smooth = 1.0
    
    probs = F.sigmoid(logits)
    m1 = probs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    score = 1 - score.sum() / num
    
    return score

def IouSemanticAwareLoss(pred_masks, first_gt_mask):
    """
    loss for single sound source segmentation

    IoU Loss
    """
    total_loss = 0
    f1_iou_loss = F1_IoU_BCELoss(pred_masks, first_gt_mask)#*1.5
    total_loss += f1_iou_loss
    dice_loss = DiceLoss(pred_masks, first_gt_mask)
    total_loss += dice_loss

    loss_dict = {}
    loss_dict['iou_loss'] = f1_iou_loss.item()
    loss_dict['dice_loss'] = dice_loss.item()

    return total_loss, loss_dict


if __name__ == "__main__":

    pdb.set_trace()
