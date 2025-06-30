import sys
import numpy as np
import torch
from torch.autograd import Function
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

sys.path.append(".")

def iou(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.mean()

class DiceCoeff(Function):

    @staticmethod
    def forward(ctx, input, target):
        eps = 0.0001
        inter = torch.dot(input.view(-1), target.view(-1))
        union = torch.sum(input) + torch.sum(target) + eps
        t = (2 * inter.float() + eps) / union.float()
        # Use clone().detach() for inter and union to avoid warnings
        ctx.save_for_backward(input, target, inter.clone().detach(), union.clone().detach())
        return t

    @staticmethod
    def backward(ctx, grad_output):
        input, target, inter, union = ctx.saved_tensors
        grad_input = grad_target = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * union - inter) / (union * union)
        if ctx.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    s = torch.FloatTensor(1).to(device=input.device).zero_()
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff.apply(c[0], c[1]) 

    return s / (i + 1)

def calculate_metrics(pred, gt):
    SMOOTH = 1e-6
    pred = pred > 0.5
    gt = gt > 0.5 

    tp = (pred & gt).sum().float()
    tn = ((~pred) & (~gt)).sum().float()
    fp = (pred & (~gt)).sum().float()
    fn = ((~pred) & gt).sum().float()

    accuracy = (tp + tn) / (tp + tn + fp + fn + SMOOTH)
    sensitivity = tp / (tp + fn + SMOOTH)
    specificity = tn / (tn + fp + SMOOTH)
    f1_score = 2 * tp / (2 * tp + fp + fn + SMOOTH)

    return accuracy.item(), sensitivity.item(), specificity.item(), f1_score.item()

def eval_seg(pred, true_mask_p, threshold=(0.1, 0.3, 0.5, 0.7, 0.9)):
    b, c, h, w = pred.size()
    iou_d, iou_c, disc_dice, cup_dice = 0, 0, 0, 0
    accuracy, sensitivity, specificity, f1_score = 0, 0, 0, 0

    for th in threshold:
        gt_vmask_p = (true_mask_p > th).float()
        vpred = (pred > th).float()
        vpred_cpu = vpred.cpu()
        
        disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
        disc_mask = gt_vmask_p[:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
        
        iou_d += iou(disc_pred, disc_mask)
        disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()

        # Calculate additional metrics
        accuracy_, sensitivity_, specificity_, f1_score_ = calculate_metrics(vpred[:,0,:,:], gt_vmask_p[:,0,:,:])
        accuracy += accuracy_
        sensitivity += sensitivity_
        specificity += specificity_
        f1_score += f1_score_

    # Averaging over thresholds
    iou_d /= len(threshold)
    disc_dice /= len(threshold)
    accuracy /= len(threshold)
    sensitivity /= len(threshold)
    specificity /= len(threshold)
    f1_score /= len(threshold)

    return iou_d, disc_dice, accuracy, sensitivity, specificity, f1_score

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--inp_pth")
    argParser.add_argument("--out_pth")
    args = argParser.parse_args()
    
    mix_res = (0, 0, 0, 0, 0, 0)
    num = 0
    pred_path = args.inp_pth
    gt_path = args.out_pth
    
    for root, dirs, files in os.walk(pred_path, topdown=False):
        for name in files:
            if 'ens' in name:
                num += 1
                ind = name.split('_')[0]
                pred = Image.open(os.path.join(root, name)).convert('L')
                gt_name = "ISIC_" + ind + "_Segmentation.png"
                gt = Image.open(os.path.join(gt_path, gt_name)).convert('L')

                pred = transforms.PILToTensor()(pred)
                pred = torch.unsqueeze(pred, 0).float() / pred.max()
                gt = transforms.PILToTensor()(gt)
                gt = transforms.Resize((256,256))(gt)
                gt = torch.unsqueeze(gt, 0).float() / 255.0

                temp = eval_seg(pred, gt)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

    avg_metrics = tuple([a / num for a in mix_res])
    iou, dice, accuracy, sensitivity, specificity, f1_score = avg_metrics

    print(f"IoU: {iou}")
    print(f"Dice Coefficient: {dice}")
    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"F1 Score: {f1_score}")

if __name__ == "__main__":
    main()
