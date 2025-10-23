import math, os, time, torch
from .checkpoint import *
from .evaluation import *

# if started from jupter use tqdm.notebook else use nromal tqdm
from tqdm.auto import tqdm
import wandb

def train_pnn(nn, train_loader, valid_loader, lossfunction, optimizer, args, logger, UUID='default'):
    start_training_time = time.time()
    evaluator = Evaluator(args)
    
    best_valid_loss = math.inf
    patience = 0
    early_stop = False

    # Try loading checkpoint
    ckpt = load_checkpoint(UUID, args.temppath)
    if ckpt:
        current_epoch, nn, optimizer, best_valid_loss = ckpt
        logger.info(f'Restart previous training from epoch {current_epoch}')
        print(f'Restart previous training from epoch {current_epoch}')
    else:
        current_epoch = 0

    # Epoch loop with tqdm
    epoch_bar = tqdm(range(current_epoch, 10**10), desc="Training Progress", position=0)

    for epoch in epoch_bar:
        start_epoch_time = time.time()
        total_train_loss = 0.0
        total_train_samples = 0

        # tqdm for training batches
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False, position=1)
        for x_train, y_train in train_bar:
            L_train_batch = lossfunction(nn, x_train, y_train)
            train_acc, train_power = evaluator(nn, x_train, y_train)

            optimizer.zero_grad()
            L_train_batch.backward()
            optimizer.step()

            batch_size = x_train.size(0)
            total_train_loss += L_train_batch.item() * batch_size
            total_train_samples += batch_size

            # Update tqdm postfix
            train_bar.set_postfix({
                "loss": f"{L_train_batch.item():.4e}",
                "acc": f"{train_acc:.4f}",
                "power": f"{train_power.item():.2e}"
            })

        L_train = total_train_loss / total_train_samples

        # Validation loop
        total_val_loss = 0.0
        total_val_samples = 0
        with torch.no_grad():
            val_bar = tqdm(valid_loader, desc=f"Epoch {epoch} [Valid]", leave=False, position=2)
            for x_val, y_val in val_bar:
                L_val_batch = lossfunction(nn, x_val, y_val)
                valid_acc, valid_power = evaluator(nn, x_val, y_val)
                batch_size = x_val.size(0)
                total_val_loss += L_val_batch.item() * batch_size
                total_val_samples += batch_size

                val_bar.set_postfix({
                    "loss": f"{L_val_batch.item():.4e}",
                    "acc": f"{valid_acc:.4f}"
                })

        L_valid = total_val_loss / total_val_samples

        wandb.log({
            "epoch": epoch,
            "train_loss": L_train,
            "train_acc": train_acc,
            "train_power": train_power,
            "val_loss": L_valid,
            "val_acc": valid_acc,
            "val_power": valid_power,
            "lr": current_lr,
            "patience": patience_lr,
        })

        if args.recording:
            record_checkpoint(epoch, nn, L_train, L_valid, UUID, args.recordpath)

        # Early stopping logic
        if L_valid < best_valid_loss:
            best_valid_loss = L_valid
            save_checkpoint(epoch, nn, optimizer, best_valid_loss, UUID, args.temppath)
            patience = 0
        else:
            patience += 1

        # Update outer tqdm description
        epoch_bar.set_postfix({
            "TrainLoss": f"{L_train:.4e}",
            "ValidLoss": f"{L_valid:.4e}",
            "Patience": patience
        })

        if patience > args.PATIENCE:
            print('Early stop.')
            logger.info('Early stop.')
            early_stop = True
            break

        if (time.time() - start_training_time) >= args.TIMELIMITATION * 60 * 60:
            print('Time limitation reached.')
            logger.warning('Time limitation reached.')
            break

        if not epoch % args.report_freq:
            msg = (f"| Epoch: {epoch:-6d} | Train loss: {L_train:.4e} | Valid loss: {L_valid:.4e} | "
                   f"Train acc: {train_acc:.4f} | Valid acc: {valid_acc:.4f} | patience: {patience:-3d} | "
                   f"Epoch time: {time.time() - start_epoch_time:.1f}s | Power: {train_power.item():.2e} |")
            print(msg)
            logger.info(msg)

    _, resulted_nn, _, _ = load_checkpoint(UUID, args.temppath)
    
    if early_stop:
        os.remove(f'{args.temppath}/{UUID}.ckp')

    epoch_bar.close()
    return resulted_nn, early_stop



def train_pnn_progressive(nn, train_loader, valid_loader, lossfunction, optimizer, args, logger, UUID='default'):
    start_training_time = time.time()
    evaluator = Evaluator(args)

    best_valid_loss = math.inf
    current_lr = args.LR
    patience_lr = 0
    lr_update = False
    early_stop = False

    # Try loading checkpoint
    ckpt = load_checkpoint(UUID, args.temppath)
    if ckpt:
        current_epoch, nn, optimizer, best_valid_loss = ckpt
        for g in optimizer.param_groups:
            current_lr = g['lr']
            g['params'] = nn.GetParam()
        logger.info(f'Restart previous training from epoch {current_epoch} with lr: {current_lr}.')
        print(f'Restart previous training from epoch {current_epoch} with lr: {current_lr}.')
    else:
        current_epoch = 0

    # Outer tqdm for epochs (auto backend avoids widget errors)
    epoch_bar = tqdm(range(current_epoch, 10**10), desc="Progressive Training", position=0)

    for epoch in epoch_bar:
        start_epoch_time = time.time()
        total_train_loss = 0.0
        total_train_samples = 0
        total_train_acc = 0.0
        total_train_power = 0.0

        # tqdm for training batches
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False, position=1)
        for batch_idx, (x_train, y_train) in enumerate(train_bar):
            L_train_batch = lossfunction(nn, x_train, y_train)
            train_acc, train_power = evaluator(nn, x_train, y_train)

            optimizer.zero_grad()
            L_train_batch.backward()

            if epoch % 5 == 0 and batch_idx == 0:  # log every few epochs to save time
                for name, param in nn.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())})

            optimizer.step()

            batch_size = x_train.size(0)
            total_train_loss += L_train_batch.item() * batch_size
            total_train_samples += batch_size
            total_train_acc += train_acc * batch_size
            total_train_power += train_power * batch_size

            train_bar.set_postfix({
                "loss": f"{L_train_batch.item():.4e}",
                "acc": f"{train_acc:.4f}",
                "lr": f"{current_lr:.2e}"
            })
        train_bar.close()

        L_train = total_train_loss / total_train_samples
        train_acc = total_train_acc / total_train_samples
        train_power = total_train_power / total_train_samples

        # Validation phase with tqdm
        total_val_loss = 0.0
        total_val_samples = 0
        total_val_acc = 0.0
        total_val_power = 0.0
        with torch.no_grad():
            val_bar = tqdm(valid_loader, desc=f"Epoch {epoch} [Valid]", leave=False, position=2)
            for x_val, y_val in val_bar:
                L_val_batch = lossfunction(nn, x_val, y_val)
                valid_acc, valid_power = evaluator(nn, x_val, y_val)
                batch_size = x_val.size(0)

                total_val_loss += L_val_batch.item() * batch_size
                total_val_samples += batch_size
                total_val_acc += valid_acc * batch_size
                total_val_power += valid_power * batch_size

                val_bar.set_postfix({
                    "loss": f"{L_val_batch.item():.4e}",
                    "acc": f"{valid_acc:.4f}"
                })
            val_bar.close()

        L_valid = total_val_loss / total_val_samples
        valid_acc = total_val_acc / total_val_samples
        valid_power = total_val_power / total_val_samples

        wandb.log({
            "epoch": epoch,
            "train_loss": L_train,
            "train_acc": train_acc,
            "train_power": train_power,
            "val_loss": L_valid,
            "val_acc": valid_acc,
            "val_power": valid_power,
            "lr": current_lr,
            "patience": patience_lr,
        })

        # Logging and checkpointing
        if args.recording:
            record_checkpoint(epoch, nn, L_train, L_valid, UUID, args.recordpath)

        if L_valid < best_valid_loss:
            best_valid_loss = L_valid
            save_checkpoint(epoch, nn, optimizer, best_valid_loss, UUID, args.temppath)
            patience_lr = 0
        else:
            patience_lr += 1

        if patience_lr > args.LR_PATIENCE:
            print('LR update triggered.')
            lr_update = True

        if lr_update:
            lr_update = False
            patience_lr = 0
            _, nn, _, _ = load_checkpoint(UUID, args.temppath)
            logger.info('Loaded best network to warm start training with lower LR.')
            for g in optimizer.param_groups:
                g['params'] = nn.GetParam()
                g['lr'] = g['lr'] * args.LR_DECAY
                current_lr = g['lr']
            logger.info(f'LR updated to {current_lr}.')
            print(f'Learning rate decayed to {current_lr:.2e}')

        if current_lr < args.LR_MIN:
            print('Early stop due to LR below minimum.')
            logger.info('Early stop due to LR below minimum.')
            early_stop = True
            break

        # Epoch timing and stopping
        end_epoch_time = time.time()
        if (end_epoch_time - start_training_time) >= args.TIMELIMITATION * 60 * 60:
            print('Time limitation reached.')
            logger.warning('Time limitation reached.')
            break

        # Update outer progress bar
        epoch_bar.set_postfix({
            "TrainLoss": f"{L_train:.4e}",
            "ValidLoss": f"{L_valid:.4e}",
            "LR": f"{current_lr:.2e}",
            "Patience": patience_lr
        })


        # Reporting
        if not epoch % args.report_freq:
            msg = (f"| Epoch: {epoch:-6d} | Train loss: {L_train:.4e} | Valid loss: {L_valid:.4e} | "
                   f"Train acc: {train_acc:.4f} | Valid acc: {valid_acc:.4f} | Patience: {patience_lr:-3d} | "
                   f"LR: {current_lr:.2e} | Epoch time: {end_epoch_time - start_epoch_time:.1f}s | "
                   f"Power: {train_power.item():.2e} |")
            logger.info(msg)

    # Load best model and clean up
    _, resulted_nn, _, _ = load_checkpoint(UUID, args.temppath)
    if early_stop:
        try:
            os.remove(f'{args.temppath}/{UUID}.ckp')
        except OSError:
            pass

    epoch_bar.close()
    return resulted_nn, early_stop
