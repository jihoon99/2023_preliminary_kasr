import torch
import numpy as np
import math
from dataclasses import dataclass
import time
from nova import DATASET_PATH


def training(config, dataloader, optimizer, model, criterion, metric, train_begin_time, device):

    model.train()

    log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] TRAINING Start')
    epoch_begin_time = time.time()
    cnt = 0
    for inputs, targets, input_lengths, target_lengths in dataloader: # input_lengths : audio seq length, target_length : token length
        begin_time = time.time()

        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        # model = model.to(device) # 모델을 불러 올 때 이미 gpu에 올림.

        outputs, output_lengths = model(inputs, input_lengths)

        loss = criterion(
            outputs.transpose(0, 1),
            targets[:, 1:],
            tuple(output_lengths),
            tuple(target_lengths)
        )

        y_hats = outputs.max(-1)[1]

        # batch 128 크다 그러니까 : cumulate backward step 방법론 생각해봄직함. 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(model)

        total_num += int(input_lengths.sum())
        epoch_loss_total += loss.item()

        torch.cuda.empty_cache()

        if cnt % config.print_every == 0:

            current_time = time.time()
            elapsed = current_time - begin_time
            epoch_elapsed = (current_time - epoch_begin_time) / 60.0    # 아 초, 단위로 한거구나.
            train_elapsed = (current_time - train_begin_time) / 3600.0  # 시간 단위로 변환한거구나.
            cer = metric(targets[:, 1:], y_hats)
            print(log_format.format(
                cnt, len(dataloader), loss,
                cer, elapsed, epoch_elapsed, train_elapsed,
                optimizer.get_lr(),
            ))
        cnt += 1
    return model, epoch_loss_total/len(dataloader), metric(targets[:, 1:], y_hats)


def validating(mode, config, dataloader, optimizer, model, criterion, metric, train_begin_time, device):

    model.eval()


    log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] VALIDATING Start')
    epoch_begin_time = time.time()
    cnt = 0

    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in dataloader: # input_lengths : audio seq length, target_length : token length
            begin_time = time.time()

            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = torch.as_tensor(target_lengths).to(device)
            # model = model.to(device) # 모델을 불러 올 때 이미 gpu에 올림.

            outputs, output_lengths = model(inputs, input_lengths)

            loss = criterion(
                outputs.transpose(0, 1),
                targets[:, 1:],
                tuple(output_lengths),
                tuple(target_lengths)
            )

            y_hats = outputs.max(-1)[1]

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(model)

            total_num += int(input_lengths.sum())
            epoch_loss_total += loss.item()

            torch.cuda.empty_cache()

            if cnt % config.print_every == 0:

                current_time = time.time()
                elapsed = current_time - begin_time
                epoch_elapsed = (current_time - epoch_begin_time) / 60.0
                train_elapsed = (current_time - train_begin_time) / 3600.0
                cer = metric(targets[:, 1:], y_hats)
                print(log_format.format(
                    cnt, len(dataloader), loss,
                    cer, elapsed, epoch_elapsed, train_elapsed,
                    optimizer.get_lr(),
                ))
            cnt += 1
        return model, epoch_loss_total/len(dataloader), metric(targets[:, 1:], y_hats)
    

def decoder():
    '''
    LM decoder, 혹은 그냥 decoder 구현 필요성이 있음.
    '''
    pass