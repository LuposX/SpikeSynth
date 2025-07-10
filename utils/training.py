import time
import math
from .checkpoint import *
from .evaluation import *

def train_pnn(nn, train_loader, valid_loader, lossfunction, optimizer, args, logger, UUID='default'):
    start_training_time = time.time()
    
    evaluator = Evaluator(args)
    
    best_valid_loss = math.inf
    patience = 0
    
    early_stop = False
    
    if load_checkpoint(UUID, args.temppath):
        current_epoch, nn, optimizer, best_valid_loss = load_checkpoint(UUID, args.temppath)
        logger.info(f'Restart previous training from {current_epoch} epoch')
        print(f'Restart previous training from {current_epoch} epoch')
    else:
        current_epoch = 0
        
    for epoch in range(current_epoch, 10**10):
        start_epoch_time = time.time()
        
        msg = ''
        
        total_train_loss = 0.0
        total_train_samples = 0
        for x_train, y_train in train_loader:
            msg += f'{current_lr}'
            msg += f'Hyperparameters in printed neural network for training :\n Epoch : {epoch:-6d} |\n'

            L_train_batch = lossfunction(nn, x_train, y_train)
            train_acc, train_power = evaluator(nn, x_train, y_train)

            optimizer.zero_grad()
            L_train_batch.backward()
            optimizer.step()

            batch_size = x_train.size(0)
            total_train_loss += L_train_batch.item() * batch_size
            total_train_samples += batch_size

        L_train = total_train_loss / total_train_samples

        with torch.no_grad():
            total_val_loss = 0.0
            total_val_samples = 0
            for x_val, y_val in valid_loader:  
                L_val_batch = lossfunction(nn, x_val, y_val)
                valid_acc, valid_power = evaluator(nn, x_val, y_val)

                batch_size = x_val.size(0)
                total_val_loss += L_val_batch.item() * batch_size
                total_val_samples += batch_size
                
            L_valid = total_val_loss / total_val_samples
        
        logger.debug(msg)
        
        if args.recording:
            record_checkpoint(epoch, nn, L_train, L_valid, UUID, args.recordpath)
            
        if L_valid < best_valid_loss:
            best_valid_loss = L_valid
            save_checkpoint(epoch, nn, optimizer, best_valid_loss, UUID, args.temppath)
            patience = 0
        else:
            patience += 1

        if patience > args.PATIENCE:
            print('Early stop.')
            logger.info('Early stop.')
            early_stop = True
            break
        
        end_epoch_time = time.time()
        end_training_time = time.time()
        if (end_training_time - start_training_time) >= args.TIMELIMITATION*60*60:
            print('Time limination reached.')
            logger.warning('Time limination reached.')
            break
        
        if not epoch % args.report_freq:
            print(f'| Epoch: {epoch:-6d} | Train loss: {L_train:.4e} | Valid loss: {L_valid:.4e} | Train acc: {train_acc:.4f} | Valid acc: {valid_acc:.4f} | patience: {patience:-3d} | Epoch time: {end_epoch_time-start_epoch_time:.1f} | Power: {train_power.item():.2e} |')
            logger.info(f'| Epoch: {epoch:-6d} | Train loss: {L_train:.4e} | Valid loss: {L_valid:.4e} | Train acc: {train_acc:.4f} | Valid acc: {valid_acc:.4f} | patience: {patience:-3d} | Epoch time: {end_epoch_time-start_epoch_time:.1f} | Power: {train_power.item():.2e} |')
        
    _, resulted_nn, _,_ = load_checkpoint(UUID, args.temppath)
    
    if early_stop:
        os.remove(f'{args.temppath}/{UUID}.ckp')

    return resulted_nn, early_stop


def train_pnn_progressive(nn, train_loader, valid_loader, lossfunction, optimizer, args, logger, UUID='default'):
    start_training_time = time.time()
    
    evaluator = Evaluator(args)
    
    best_valid_loss = math.inf
    current_lr = args.LR
    patience_lr = 0
    
    lr_update = False
    early_stop = False
    
    if load_checkpoint(UUID, args.temppath):
        current_epoch, nn, optimizer, best_valid_loss = load_checkpoint(UUID, args.temppath)
        for g in optimizer.param_groups:
            current_lr = g['lr']
            # g['params'] = [p for p in nn.parameters()]
            g['params'] = nn.GetParam()
        logger.info(f'Restart previous training from {current_epoch} epoch with lr: {current_lr}.')
        print(f'Restart previous training from {current_epoch} epoch with lr: {current_lr}.')
    else:
        current_epoch = 0

        
    for epoch in range(current_epoch, 10**10):
        start_epoch_time = time.time()
        
        msg = ''
        
        total_train_loss = 0.0
        total_train_samples = 0
        total_train_acc = 0.0
        total_train_power = 0.0
        for x_train, y_train in train_loader:
            msg += f'{current_lr}'
            msg += f'Hyperparameters in printed neural network for training :\n Epoch : {epoch:-6d} |\n'

            L_train_batch = lossfunction(nn, x_train, y_train)
            train_acc, train_power = evaluator(nn, x_train, y_train)

            optimizer.zero_grad()
            L_train_batch.backward()
            optimizer.step()

            batch_size = x_train.size(0)
            total_train_loss += L_train_batch.item() * batch_size
            total_train_samples += batch_size

            # Update the total train_acc and train_power
            total_train_acc += train_acc * batch_size
            total_train_power += train_power * batch_size

        L_train = total_train_loss / total_train_samples
        train_acc = total_train_acc / total_train_samples
        train_power = total_train_power / total_train_samples

        with torch.no_grad():
            total_val_loss = 0.0
            total_val_samples = 0
            total_val_acc = 0.0
            total_val_power = 0.0

            for x_val, y_val in valid_loader:
                L_val_batch = lossfunction(nn, x_val, y_val)
                valid_acc, valid_power = evaluator(nn, x_val, y_val)

                batch_size = x_val.size(0)
                total_val_loss += L_val_batch.item() * batch_size
                total_val_samples += batch_size

                # Update the total valid_acc and valid_power
                total_val_acc += valid_acc * batch_size
                total_val_power += valid_power * batch_size

            L_valid = total_val_loss / total_val_samples
            valid_acc = total_val_acc / total_val_samples
            valid_power = total_val_power / total_val_samples

        
        logger.debug(msg)
        
        if args.recording:
            record_checkpoint(epoch, nn, L_train, L_valid, UUID, args.recordpath)
            
        if L_valid < best_valid_loss:
            best_valid_loss = L_valid
            save_checkpoint(epoch, nn, optimizer, best_valid_loss, UUID, args.temppath)
            patience_lr = 0
        else:
            patience_lr += 1

        if patience_lr > args.LR_PATIENCE:
            print('lr update')
            lr_update = True
        
        if lr_update:
            lr_update = False
            patience_lr = 0
            _, nn, _,_ = load_checkpoint(UUID, args.temppath)
            logger.info('load best network to warm start training with lower lr.')
            for g in optimizer.param_groups:
                # g['params'] = [p for p in nn.parameters()]
                g['params'] = nn.GetParam()
                g['lr'] = g['lr'] * args.LR_DECAY
                current_lr = g['lr']
            logger.info(f'lr update to {current_lr}.')

        if current_lr < args.LR_MIN:
            early_stop = True
            print('early stop.')
            logger.info('Early stop.')
            break
        
        end_epoch_time = time.time()
        end_training_time = time.time()
        if (end_training_time - start_training_time) >= args.TIMELIMITATION*60*60:
            print('Time limination reached.')
            logger.warning('Time limination reached.')
            break
        
        if not epoch % args.report_freq:
            print(f'| Epoch: {epoch:-6d} | Train loss: {L_train:.4e} | Valid loss: {L_valid:.4e} | Train acc: {train_acc:.4f} | Valid acc: {valid_acc:.4f} | patience: {patience_lr:-3d} | lr: {current_lr} | Epoch time: {end_epoch_time-start_epoch_time:.1f} | Power: {train_power.item():.2e} |')
            logger.info(f'| Epoch: {epoch:-6d} | Train loss: {L_train:.4e} | Valid loss: {L_valid:.4e} | Train acc: {train_acc:.4f} | Valid acc: {valid_acc:.4f} | patience: {patience_lr:-3d} | lr: {current_lr} | Epoch time: {end_epoch_time-start_epoch_time:.1f} | Power: {train_power.item():.2e} |')
        
    _, resulted_nn, _,_ = load_checkpoint(UUID, args.temppath)
    
    if early_stop:
        os.remove(f'{args.temppath}/{UUID}.ckp')

    return resulted_nn, early_stop
