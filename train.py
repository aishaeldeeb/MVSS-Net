import argparse
import os
from sys import platform

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from models.mvssnet import get_mvss

#??
def dice_loss(gt, out, smooth = 1.0): 
    gt = gt.view(-1) #?
    out = out.view(-1) #?

    intersection = (gt * out).sum() 
    dice = (2.0 * intersection + smooth) / (torch.square(gt).sum() + torch.square(out).sum() + smooth) # TODO: need to confirm this matches what the paper says, and also the calculation/result is correct

    return 1.0 - dice

def bgr_to_rgb(t):
    b, g, r = torch.unbind(t, 1) #?
    return torch.stack((r, g, b), 1)#?

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--train_paths_file", type=str, default="/dataset/files.txt", help="path to the file with input paths") # each line of this file should contain "/path/to/image.ext /path/to/mask.ext /path/to/edge.ext 1 (for fake)/0 (for real)"; for real image.ext, set /path/to/mask.ext and /path/to/edge.ext as a string None
    parser.add_argument("--valid_paths_file", type=str, default="/dataset/files.txt", help="path to the file with input paths") # each line of this file should contain "/path/to/image.ext /path/to/mask.ext /path/to/edge.ext 1 (for fake)/0 (for real)"; for real image.ext, set /path/to/mask.ext and /path/to/edge.ext as a string None
    parser.add_argument("--image_size", type=int, default=512, help="size of the images")
    parser.add_argument("--batch_size", type=int, default=12, help="size of the batches") # no default value given by paper
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--workers", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument('--decay_epoch', type=int, default=50, help='decay')
    parser.add_argument("--lambda_seg", type=float, default=0.16, help="pixel-scale loss weight (alpha)")
    parser.add_argument("--lambda_clf", type=float, default=0.04, help="image-scale loss weight (beta)")
    parser.add_argument("--run_name", type=str, default="MVSS-Net", help="run name")
    parser.add_argument("--log_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="batch interval between model checkpoints")
    parser.add_argument('--load_path', type=str, help='pretrained model or checkpoint for continued training')
    parser.add_argument('--nGPU', type=int, default=1, help='number of gpus') # TODO: multiple GPU support
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = get_mvss(backbone='resnet50',
                         pretrained_base=True,
                         nclass=1,
                         sobel=True,
                         constrain=True,
                         n_input=args.channels,
                         ).to(device)

    # Losses that are built-in in PyTorch
    criterion_clf = nn.BCEWithLogitsLoss().to(device)

    # Load pretrained models
    # if args.load_path != None:
    #     print('Load pretrained model: ' + args.load_path)
    #     model.load_state_dict(torch.load(args.load_path))

    # Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # Time for log
    logtm = datetime.now().strftime("%Y%m%d%H%M%S")

    # Dataset
    train_dataset = Datasets(args.train_paths_file, args.image_size)
    valid_dataset= Datasets(args.valid_paths_file, args.image_size)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=args.batch_size)
    # Conversion from epoch to step/iter
    decay_iter = args.decay_epoch * len(train_dataloader)

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                                step_size=decay_iter,
                                                                gamma=0.5)

    # ----------
    #  Training
    # ----------
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter("logs/" + logtm + "_" + args.run_name)
    checkpoint_dir = "checkpoints/" + logtm + "_" + args.run_name
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(args.epoch, args.n_epochs):
        print("Starting Epoch ", epoch + 1)

        # loss sum for epoch
        epoch_total_seg = 0
        epoch_total_clf = 0
        epoch_total_edg = 0

        epoch_total_model = 0

        val_epoch_total_seg = 0
        val_epoch_total_clf = 0
        val_epoch_total_edg = 0

        val_epoch_total_model = 0
    

        epoch_val_loss = 0
        epoch_steps = 0
        val_epoch_steps = 0
        for step, data in enumerate(train_dataloader):
            curr_steps = epoch * len(train_dataloader) + step + 1

            # Read from dataloader
            in_imgs = Variable(data["input"].type(Tensor))
            in_masks = Variable(data["mask"].type(Tensor))
            in_edges = Variable(data["edge"].type(Tensor))
            in_labels = Variable(data["label"].type(Tensor))

            # ------------------
            #  Train Generators
            # ------------------

            optimizer.zero_grad()

            # Prediction
            out_edges, out_masks = model(in_imgs)
            out_edges = torch.sigmoid(out_edges)
            out_masks = torch.sigmoid(out_masks)

            # Pixel-scale loss
            loss_seg = dice_loss(in_masks, out_masks)

            # Edge loss
            # TODO: is it the same as the paper?
            loss_edg = dice_loss(in_edges, out_edges)

            # Image-scale loss (with GMP)
            # TODO: GeM from MVSS-Net++
            gmp = nn.MaxPool2d(args.image_size)
            out_labels = gmp(out_masks).squeeze()
            loss_clf = criterion_clf(in_labels, out_labels)

            # Total loss
            alpha = args.lambda_seg
            beta = args.lambda_clf

            weighted_loss_seg = alpha * loss_seg
            weighted_loss_clf = beta * loss_clf
            weighted_loss_edg = (1.0 - alpha - beta) * loss_edg

            loss = weighted_loss_seg + weighted_loss_clf + weighted_loss_edg

            # backward prop and step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # log losses for epoch
            epoch_steps += 1

            epoch_total_seg += weighted_loss_seg
            epoch_total_clf += weighted_loss_clf
            epoch_total_edg += weighted_loss_edg
            epoch_total_model += loss
            
            # --------------
            #  Log Progress (for certain steps)
            # --------------
            if step % args.log_interval == 0:
                print(f"[Epoch {epoch + 1}/{args.n_epochs}] [Batch {step + 1}/{len(train_dataloader)}] "
                    f"[Total Loss {loss:.3f}]"
                    f"[Pixel-scale Loss {weighted_loss_seg:.3e}]"
                    f"[Edge Loss {weighted_loss_edg:.3e}]"
                    f"[Image-scale Loss {weighted_loss_clf:.3e}]"
                    f"")

                writer.add_scalar("LearningRate", lr_scheduler.get_last_lr()[0],
                                curr_steps)
                writer.add_scalar("Loss/Total Loss", loss, epoch * len(train_dataloader) + step)
                writer.add_scalar("Loss/Pixel-scale", weighted_loss_seg, curr_steps)
                writer.add_scalar("Loss/Edge", weighted_loss_edg, curr_steps)
                writer.add_scalar("Loss/Image-scale", weighted_loss_clf, curr_steps)

                in_imgs_rgb = bgr_to_rgb(in_imgs.clone().detach())
                writer.add_images('Input Img', in_imgs_rgb, epoch * len(train_dataloader) + step)

                writer.add_images('Input Mask', in_masks, epoch * len(train_dataloader) + step)
                writer.add_images('Output Mask', out_masks, epoch * len(train_dataloader) + step)
                writer.add_images('Input Edge', in_edges, epoch * len(train_dataloader) + step)
                writer.add_images('Output Edge', out_edges, epoch * len(train_dataloader) + step)

            # save model parameters
            # TODO: you can change when the parameters are saved
            if step % args.checkpoint_interval == 0:
                tm = datetime.now().strftime("%Y%m%d%H%M%S")
                torch.save(model.state_dict(),
                        os.path.join(checkpoint_dir, tm + '_' + args.run_name + '_' + str(
                            epoch + 1) + "_" + str(step + 1) + '.pth'))

        # ------------------
        #  Validation
        # ------------------
        if (val_dataloader):
            model.eval()

            for step, data in enumerate(val_dataloader):
                # 
                with torch.no_grad():
                    # Read from dataloader
                    # in_imgs_val, in_labels = data
                    # Read from dataloader
                    val_in_imgs= Variable(data["input"].type(Tensor))
                    val_in_masks = Variable(data["mask"].type(Tensor))
                    val_in_edges = Variable(data["edge"].type(Tensor))
                    val_in_labels = Variable(data["label"].type(Tensor))

                    # val_in_imgs = val_in_imgs.to('cuda', non_blocking=True)
                    # val_in_masks = val_in_masks.to('cuda', non_blocking=True).float().view(val_in_imgs.size(0), 1)
                    # val_in_edges = val_in_edges.to('cuda', non_blocking=True).float().view(val_in_imgs.size(0), 1)
                    # val_in_labels = val_in_labels.to('cuda', non_blocking=True).float().view(val_in_imgs.size(0), 1)

                    # out = model(in_imgs_val)
                    # Prediction
                    val_out_edges, val_out_masks = model(val_in_imgs)
                    val_out_edges = torch.sigmoid(val_out_edges)
                    val_out_masks = torch.sigmoid(val_out_masks)

                    # loss = criterion_clf(out, in_labels)
            
                    # Pixel-scale loss
                    val_loss_seg = dice_loss(val_in_masks, val_out_masks)

                    # Edge loss
                    # TODO: is it the same as the paper?
                    val_loss_edg = dice_loss(val_in_edges, val_out_edges)

                    # Image-scale loss (with GMP)
                    # TODO: GeM from MVSS-Net++
                    gmp = nn.MaxPool2d(args.image_size)
                    val_out_labels = gmp(val_out_masks).squeeze()
                    val_loss_clf = criterion_clf(val_in_labels, val_out_labels)
                    # Total loss
                    alpha = args.lambda_seg
                    beta = args.lambda_clf

                    val_weighted_loss_seg = alpha * val_loss_seg
                    val_weighted_loss_clf = beta * val_loss_clf
                    val_weighted_loss_edg = (1.0 - alpha - beta) * val_loss_edg

                    val_loss = val_weighted_loss_seg + val_weighted_loss_clf + val_weighted_loss_edg

                    val_epoch_total_seg += val_weighted_loss_seg
                    val_epoch_total_clf += val_weighted_loss_clf
                    val_epoch_total_edg += val_weighted_loss_edg
                    val_epoch_total_model += val_loss

            # early 
            best_val_loss = float('inf')
            n_last_epochs = 0
            early_stopping = False
            if val_epoch_total_model <= best_val_loss:
                best_val_loss = epoch_val_loss
                n_last_epochs = 0
            else:
                n_last_epochs += 1

                if (n_last_epochs >= args.n_early):
                    early_stopping = True

            # log losses for epoch
            val_epoch_steps += 1

            
            
        # --------------
        #  Log Progress (for epoch)
        # --------------
        # loss average for epoch
        if (epoch_steps != 0):
            epoch_avg_seg = epoch_total_seg / epoch_steps
            epoch_avg_edg = epoch_total_edg / epoch_steps
            epoch_avg_clf = epoch_total_clf / epoch_steps
            epoch_avg_model = epoch_total_model / epoch_steps

            print(f"[Epoch {epoch + 1}/{args.n_epochs}]"
                    f"[Epoch Total Loss {epoch_avg_model:.3f}]"
                    f"[Epoch Pixel-scale Loss {epoch_avg_seg:.3e}]"
                    f"[Epoch Edge Loss {epoch_avg_edg:.3e}]"
                    f"[Epoch Image-scale Loss {epoch_avg_clf:.3e}]"
                    f"")

            writer.add_scalar("Epoch LearningRate", lr_scheduler.get_last_lr()[0],
                                epoch)
            writer.add_scalar("Epoch Loss/Total Loss", epoch_avg_model, epoch)
            writer.add_scalar("Epoch Loss/Pixel-scale", epoch_avg_seg, epoch)
            writer.add_scalar("Epoch Loss/Edge", epoch_avg_edg, epoch)
            writer.add_scalar("Epoch Loss/Image-scale", epoch_avg_clf, epoch)

            in_imgs_rgb = bgr_to_rgb(in_imgs.clone().detach())

            writer.add_images('Epoch Input Img', in_imgs_rgb, epoch)

            writer.add_images('Epoch Input Mask', in_masks, epoch)
            writer.add_images('Epoch Output Mask', out_masks, epoch)
            writer.add_images('Epoch Input Edge', in_edges, epoch)
            writer.add_images('Epoch Output Edge', out_edges, epoch)

        if (val_epoch_steps != 0):
            val_epoch_avg_seg = val_epoch_total_seg / val_epoch_steps
            val_epoch_avg_edg = val_epoch_total_edg / val_epoch_steps
            val_epoch_avg_clf = val_epoch_total_clf / val_epoch_steps
            val_epoch_avg_model = val_epoch_total_model / val_epoch_steps

            # epoch_val_avg_loss = epoch_val_loss / len(val_dataloader)

            

            best_val_loss_avg = best_val_loss / len(val_dataloader)

            print(f"[Epoch {epoch + 1}/{args.n_epochs}]"
                    f"[Epoch Validation Loss {val_epoch_avg_model:.3f} (best Val Loss {best_val_loss_avg:.3f} last for {n_last_epochs:d})]"
                    f"[Epoch Validation Pixel-scale Loss {val_epoch_avg_seg:.3e}]"
                    f"[Epoch Validation Edge Loss {val_epoch_avg_edg:.3e}]"
                    f"[Epoch Validation Image-scale Loss {val_epoch_avg_clf:.3e}]"
                    f"")

            writer.add_scalar("Val Epoch Loss/Total Loss", val_epoch_avg_model, epoch)
            writer.add_scalar("Val Epoch Loss/Pixel-scale", val_epoch_avg_seg, epoch)
            writer.add_scalar("Val Epoch Loss/Edge", val_epoch_avg_edg, epoch)
            writer.add_scalar("Val Epoch Loss/Image-scale", val_epoch_avg_clf, epoch)

            val_in_imgs_rgb = bgr_to_rgb(val_in_imgs.clone().detach())

            writer.add_images('Validation Epoch Input Img', val_in_imgs_rgb, epoch)

            writer.add_images('Validation Epoch Input Mask', val_in_masks, epoch)
            writer.add_images('Validation Epoch Output Mask', val_out_masks, epoch)
            writer.add_images('Validation Epoch Input Edge', val_in_edges, epoch)
            writer.add_images('Validation Epoch Output Edge', val_out_edges, epoch)


    print("Finished training")

if platform == "win32":
    if __name__ == '__main__':
        main()
else:
    main()