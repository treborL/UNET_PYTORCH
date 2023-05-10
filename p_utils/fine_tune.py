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

import vgg_unet
from vgg_unet import vgg16bn_unet
from p_utils.data_loader import CustomDataloader
from p_utils.dice_score import dice_loss
from p_utils.evaluate import evaluate
from torch.autograd import Variable


def fine_tune_model(model: vgg_unet.VGGUnet, ft_img: str, ft_mask: str, mask_suff: str, val_percent: float,
                    batch_size: int, device: torch.device, opt: str, lr: float, weight_decay: float, amp: bool,
                    num_classes: int, num_epochs: int, gradient_clipping: float, checkpoint: bool, checkpoint_dir: str):

    model.unfreeze_pretrained()

    dataset = CustomDataloader(ft_img, ft_mask, mask_suffix=mask_suff)

    # Dataset split into train and validation
    validation_count = int(len(dataset) * (val_percent / 100.))
    train_count = len(dataset) - validation_count
    train_set, val_set = random_split(dataset, [train_count, validation_count],
                                      generator=torch.Generator().manual_seed(0))

    # Data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

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

    loss_function = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()

    global_step = 0

    print("STARTING FINE TUNING MODEL")
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

        if checkpoint:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(checkpoint_dir + 'checkpoint_epoch{}_fine_tune.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    return model