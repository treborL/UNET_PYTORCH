import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from resnet_unet import resnet_unet
from p_utils.data_loader import CustomDataloader
from p_utils.dice_score import dice_loss
from p_utils.evaluate import evaluate
from torch.autograd import Variable
from p_utils.fine_tune import fine_tune_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGG16_Unet')

    # AMP
    parser.add_argument('--amp', metavar='Mixed precision', dest='amp', type=int, default=0,
                        help='Mixed precision')

    # Checkpoint
    parser.add_argument('--checkpoint', metavar='Epoch checkpoint', dest='cp', type=int, default=1,
                        help='Save checkpoints after each epoch. Default = True')
    parser.add_argument('--checkpoint_dir', metavar='Checkpoint directory', dest='checkpoint_dir', type=str,
                        default="/checkpoints/",
                        help='Directory where checkpoints are saved')

    # Number of classes
    parser.add_argument('--num_c', metavar='Output classes count', dest='num_c', type=int, default=1,
                        help='Number of output classes')

    # Validation percent
    parser.add_argument('--val_perc', metavar='Validation percen', dest='val_perc', type=int, default=0,
                        help='percent of images to use as validation in range [0, 100]')

    # Paths
    parser.add_argument('--dir_img', metavar='Images directory', dest='dir_img', type=str, default="./dataset/input/",
                        help='path to input images')
    parser.add_argument('--dir_mask', metavar='Masks directory', dest='dir_mask', type=str,
                        default="./dataset/groundtruth/", help='path to ground truth masks')
    parser.add_argument('--mask_suf', metavar='Masks suffix', dest='mask_suff', type=str,
                        default='_mask', help='masks suffix')

    parser.add_argument('--ft_img', metavar='Fine tune images directory', dest='ft_img', type=str,
                        default="./dataset/fine_tune_input/",
                        help='fine tune images directory')
    parser.add_argument('--ft_mask', metavar='Fine tune masks directory', dest='ft_mask', type=str,
                        default="./dataset/fine_tune_groundtruth/", help='path to ground fine tune truth masks')

    # Optimization
    parser.add_argument('--lr', metavar='Learning Rate', dest='lr', type=float, default=1e-3,
                        help='learning rate of the optimization')
    parser.add_argument('--weight_decay', metavar='weight_decay', dest='weight_decay', type=float, default=1e-8,
                        help='weight decay of the optimization')
    parser.add_argument('--num_epochs', metavar='Number of epochs', dest='num_epochs', type=int, default=5,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', metavar='Minibatch size', dest='batch_size', type=int, default=1,
                        help='Number of samples per minibatch')
    parser.add_argument('--opt', metavar='Optimizer to be used', dest='opt', type=str, default='rmsprop',
                        help='adam, sgd, rmsprop')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = parser.parse_args()

    amp = True if args.amp == 1 else False
    checkpoint = True if args.cp == 1 else False
    checkpoint_dir = args.checkpoint_dir

    lr = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    # loss_function TODO
    opt = args.opt

    dir_img = args.dir_img
    dir_mask = args.dir_mask
    mask_suff = args.mask_suff
    val_percent = args.val_perc
    gradient_clipping = 1.0

    num_classes = args.num_c

    if not os.path.exists(dir_img):
        print(dir_img + " doesn't exists")  # TODO

    if not os.path.exists(dir_mask):
        print(dir_mask + " doesn't exists")  # TODO

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training params: \nLearning rate: {lr},\nWeight decay: {weight_decay},\nNumber of epochs: {num_epochs},"
          f"\nBatch size: {batch_size},\nOptimizer: {opt}")

    dataset = CustomDataloader(dir_img, dir_mask, mask_suffix=mask_suff)

    # Dataset split into train and validation
    validation_count = int(len(dataset) * (val_percent / 100.))
    train_count = len(dataset) - validation_count
    train_set, val_set = random_split(dataset, [train_count, validation_count],
                                      generator=torch.Generator().manual_seed(0))

    # Data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # Create model
    model = resnet_unet(num_classes, pretrained=True)
    model.to(device=device, memory_format=torch.channels_last)

    # Setup optimizer
    if opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    elif opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.999)
    else:
        raise f"Optimizer = {opt}, possible optimizers: adam, sgd, rmsprop."

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Losses
    # TODO
    loss_function = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()

    global_step = 0

    # Training
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=train_count, desc=f'Epoch {epoch}/{num_epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, "Image channels isn't equal to model input channels"

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = loss_function(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = loss_function(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (train_count // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)
                        logging.info('Validation Dice score: {}'.format(val_score))

        if checkpoint and epoch % 10 == 0:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(checkpoint_dir + 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    # FINE TUNE
    model = fine_tune_model(model=model, ft_img=args.ft_img, ft_mask=args.ft_mask, mask_suff=mask_suff,
                            val_percent=val_percent,
                            batch_size=batch_size, device=device, amp=amp, num_epochs=num_epochs,
                            gradient_clipping=gradient_clipping, checkpoint=checkpoint, checkpoint_dir=checkpoint_dir,
                            loss_function=loss_function, optimizer=optimizer, grad_scaler=grad_scaler,
                            scheduler=scheduler)

    # SAVE MODEL IN ONNX
    dummy_input = Variable(torch.randn(1, 3, 192, 640, device=device))

    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "bottles.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=17,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])  # the model's output names
