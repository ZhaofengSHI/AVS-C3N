import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def F5_IoU_BCELoss(pred_mask, five_gt_masks):
    """
    binary cross entropy loss (iou loss) of the total five frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    five_gt_masks: ground truth mask of the total five frames, shape: [bs*5, 1, 224, 224]
    """

    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask) # [bs*5, 1, 224, 224]
    
    # five_gt_masks = five_gt_masks.view(-1, 1, five_gt_masks.shape[-2], five_gt_masks.shape[-1]) # [bs*5, 1, 224, 224]

    # weight = torch.zeros_like(five_gt_masks).float().cuda()
    # weight = torch.fill_(weight,0.9)
    # weight[five_gt_masks>0]=1.1


    loss = nn.BCELoss()(pred_mask, five_gt_masks)

    return loss


 
def DiceLoss(logits, targets):
    num = targets.size(0)
    smooth = 1.0
    
    probs = F.sigmoid(logits)
    m1 = probs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    score = 1 - score.sum() / num
    return score



def IouSemanticAwareLoss(pred_masks, gt_mask):
    """
    loss for multiple sound source segmentation

    IoU Loss
    """
    total_loss = 0
    iou_loss = F5_IoU_BCELoss(pred_masks, gt_mask) * 1.5
    total_loss += iou_loss
    dice_loss = DiceLoss(pred_masks, gt_mask)
    total_loss += dice_loss
    

    loss_dict = {}
    loss_dict['iou_loss'] = iou_loss.item()
    loss_dict['dice_loss'] = dice_loss.item()

    return total_loss, loss_dict


if __name__ == "__main__":

    pdb.set_trace()
