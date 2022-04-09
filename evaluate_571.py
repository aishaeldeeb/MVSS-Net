import os
import cv2
import numpy as np
import sys
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay
from tqdm import tqdm
from matplotlib import pyplot as plt
import csv

import torch
import torchvision.transforms as transforms

from models.mvssnet import get_mvss

def read_paths(paths_file, subset):
    data = []

    with open(paths_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            parts = l.rstrip().split(' ')
            input_image_path = parts[0]
            mask_image_path = parts[1]
            # parts[2] is the path for edges, skipped
            label = int(parts[3])

            if (subset and subset not in input_image_path):
                continue

            data.append((input_image_path, mask_image_path, label))

    return data

def calculate_pixel_f1(pd, gt):
    # both the predition and groundtruth are empty
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        return 1.0, 0.0, 0.0
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    return f1, precision, recall

def calculate_img_score(pd, gt):
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
    acc = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos + 1e-6)
    sen = true_pos / (true_pos + false_neg + 1e-6)
    spe = true_neg / (true_neg + false_pos + 1e-6)
    f1 = 2 * sen * spe / (sen + spe)
    return acc, sen, spe, f1, true_pos, true_neg, false_pos, false_neg

def save_cm(y_true, y_pred, save_path):
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)

def save_auc(y_true, scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)

    plt.figure()
    
    plt.plot(
        fpr,
        tpr,
        label="ROC curve",
        )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument("--paths_file", type=str, default="/eval_files.txt", help="path to the file with input paths") # each line of this file should contain "/path/to/image.ext /path/to/mask.ext /path/to/edge.ext 1 (for fake)/0 (for real)"; for real image.ext, set /path/to/mask.ext and /path/to/edge.ext as a string None
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--load_path', type=str, help='path to the pretrained model', default="ckpt/mvssnet.pth")
    parser.add_argument("--image_size", type=int, default=512, help="size of the images for prediction")
    parser.add_argument("--subset", type=str, help="evaluation on certain subset")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_mvss(backbone='resnet50',
                         pretrained_base=True,
                         nclass=1,
                         sobel=True,
                         constrain=True,
                         n_input=3,
                         ).to(device)

    if os.path.exists(args.load_path):
        checkpoint = torch.load(args.load_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        print("load %s finish" % (os.path.basename(args.load_path)))
    else:
        print("%s not exist" % args.load_path)
        sys.exit()
    
    # no training
    model.eval()

    # read paths for data
    if not os.path.exists(args.paths_file):
        print("%s not exists, quit" % args.paths_file)
        sys.exit()

    if (args.subset):
        print("Evaluation on subset {}".format(args.subset))

    data = read_paths(args.paths_file, args.subset)

    print("Eval set size is {}!".format(len(data)))

    # create/reset output folder
    print("Predicted maps will be saved in :%s" % args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'masks'), exist_ok=True)

    # csv
    if (args.subset is None):
        f_csv = open(os.path.join(args.out_dir, 'pred.csv'), 'w')
        writer = csv.writer(f_csv)

        header = ['Image', 'Score', 'Pred', 'True', 'Correct']
        writer.writerow(header)

    # transforms
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()])
            
    transform_pil = transforms.Compose([transforms.ToPILImage()])

    # for storting results
    scores, labs = [], []
    f1s = [[], []]

    with torch.no_grad():
        for ix, (img_path, mask_path, lab) in enumerate(tqdm(data, mininterval = 60)):
            img = cv2.imread(img_path)
            ori_size = img.shape

            # resize to fit model
            img = transform(img).to(device).unsqueeze(0)
            
            # prediction
            _, seg = model(img)

            # resize to original
            seg = torch.sigmoid(seg).detach().cpu()
            seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]
            if len(seg) != 1:
                print("%s seg size not 1" % img_path)
                continue
            else:
                seg = seg[0].astype(np.uint8)
            seg = cv2.resize(seg, (ori_size[1], ori_size[0])) # the order of size here is important

            # save prediction
            save_seg_path = os.path.join(args.out_dir, 'masks', 'pred_' + os.path.basename(img_path).split('.')[0] + '.png')
            cv2.imwrite(save_seg_path, seg.astype(np.uint8))

            # convert from image to floating point
            seg = seg / 255.0

            score = np.max(seg)
            scores.append(score)
            labs.append(lab)
            f1 = 0
            if mask_path != 'None': # fake
                gt = cv2.imread(mask_path, 0) / 255.0
            else:
                gt = np.zeros((ori_size[0], ori_size[1]))

            if seg.shape != gt.shape:
                print("%s size not match" % img_path)
                continue

            seg = (seg > args.threshold).astype(np.float64)
            
            # pixel-level F1
            f1, _, _ = calculate_pixel_f1(seg.flatten(), gt.flatten())

            f1s[lab].append(f1)

            # write to csv
            if (args.subset is None):
                row = [img_path, score, (score > args.threshold).astype(int), lab, (score > args.threshold).astype(int) == lab]
                writer.writerow(row)

    # image-level AUC
    y_true = (np.array(labs) > args.threshold).astype(int)
    y_pred = (np.array(scores) > args.threshold).astype(int)

    if (args.subset is None):
        save_path = os.path.join(args.out_dir, 'auc.png')
        save_auc(y_true, scores, save_path)

        img_auc = roc_auc_score(y_true, scores)
    else:
        img_auc = 0.0

    meanf1 = np.mean(f1s[0] + f1s[1])
    print("pixel-f1: %.4f" % meanf1)

    acc, sen, spe, f1_imglevel, tp, tn, fp, fn = calculate_img_score(y_pred, y_true)
    print("img level acc: %.4f sen: %.4f  spe: %.4f  f1: %.4f auc: %.4f"
        % (acc, sen, spe, f1_imglevel, img_auc))
    print("combine f1: %.4f" % (2*meanf1*f1_imglevel/(f1_imglevel+meanf1+1e-6)))

    # confusion matrix
    save_path = os.path.join(args.out_dir, 'cm' + ('_' + args.subset if args.subset else '') + '.png')
    save_cm(y_true, y_pred, save_path)

    if (args.subset is None): f_csv.close()