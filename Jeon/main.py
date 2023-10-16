import torch
import queue
import os
import random
import warnings
import time
import json
import argparse
from glob import glob

from modules.preprocess import preprocessing
from modules.trainer import (
    training,
    validating,
)
from modules.utils import (
    get_optimizer,
    get_criterion,
    get_lr_scheduler,
)
# from modules.audio import (
#     FilterBankConfig,
#     MelSpectrogramConfig,
#     MfccConfig,
#     SpectrogramConfig,
# )
# from modules.model import build_model
from modules.model.deepspeech2 import build_deepspeech2
from modules.model.conformer.model import Conformer

from modules.vocab import KoreanSpeechVocabulary
from modules.data import (
    split_dataset, 
    collate_fn,
    PadFill,
    CustomPadFill,
)
from modules.utils import Optimizer
from modules.metrics import get_metric
from modules.inference import (
    single_infer,
    custom_oneToken_infer_for_testing
)

from torch.utils.data import DataLoader
import torch.nn as nn

from modules.vocab import Vocabulary

import pandas as pd
import pickle

import nova
from nova import DATASET_PATH


device = 'cuda' if torch.cuda.is_available() else 'cpu'



def save_checkpoint(checkpoint, dir):
    torch.save(checkpoint, os.path.join(dir))

def bind_model(model, parser):

    def save(dir_name, *parser):
        '''
            save trained model to nsml system
        '''
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'model_checkpoint')
        save_checkpoint(dict_for_infer['model'], save_dir)

        with open(os.path.join(dir_name, "dict_for_infer.pkl"), "wb") as f:
            pickle.dump(dict_for_infer, f)

        print(f"학습 모델 저장 완료!")


    def load(dir_name, *parser):
        '''
            load saved model from nsml system
        '''
        save_dir = os.path.join(dir_name, 'model_checkpoint')
        global checkpoint
        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint)
        global dict_for_infer
        with open(os.path.join(dir_name, "dict_for_infer.pkl"), 'rb') as f:
            dict_for_infer = pickle.load(f)
        print("    ***학습 모델 로딩 완료!")


    def infer(test_path, *parser):
        config = dict_for_infer['config']
        vocab = dict_for_infer['vocab']
        model.to('cuda')
        model.eval()

        results = []
        for i in glob(os.path.join(test_path, '*')): # glob이 도커에 안깔림...
            results.append(
                {
                    'filename': i.split('/')[-1],
                    'text': custom_oneToken_infer_for_testing(model, i, vocab, config)[0]
                }
            )
        return sorted(results, key=lambda x: x['filename'])


    nova.bind(save=save, load=load, infer=infer)  # 'nova.bind' function must be called at the end.

# def _bind_model(model, optimizer=None):
#     def save(path, *args, **kwargs):
#         state = {
#             'model': model.state_dict(),
#             'optimizer': optimizer.state_dict()
#         }
#         torch.save(state, os.path.join(path, 'model.pt'))
#         print('Model saved')

#     def load(path, *args, **kwargs):
#         state = torch.load(os.path.join(path, 'model.pt'))
#         model.load_state_dict(state['model'])
#         if 'optimizer' in state and optimizer:
#             optimizer.load_state_dict(state['optimizer'])
#         print('Model loaded')

#     # 추론
#     def infer(path, **kwargs):
#         return inference(path, model)

#     nova.bind(save=save, load=load, infer=infer)  # 'nova.bind' function must be called at the end.



# def inference(path, model, **kwargs):
#     model.eval()

#     results = []
#     # for i in [os.path.join(path,i) for i in os.listidr(path)]:
#     for i in glob(os.path.join(path, '*')): # glob이 도커에 안깔림...
#         results.append(
#             {
#                 'filename': i.split('/')[-1],
#                 'text': single_infer(model, i)[0]
#             }
#         )
#     return sorted(results, key=lambda x: x['filename'])



def spell_check(config):
    pass

def load_vocab(
        config,
        sos_id = '<s>',
        eos_id = '</s>',
        pad_id = '[pad]',
        blank_id = '<blank>',
        # unk_id = '[unk]',
        out_path = 'labels.csv'
    ):
    vocab = KoreanSpeechVocabulary(
        os.path.join(os.getcwd(), out_path), 
        output_unit='character',
        sos_id = sos_id,
        eos_id = eos_id,
        pad_id = pad_id,
        blank_id = blank_id,
        # unk_id = unk_id
        )
        
    return vocab # class it self


def build_model(
        input_size,
        config,
        vocab: Vocabulary,
        device: torch.device,
) -> nn.DataParallel:

    if config.architecture == 'deepspeech2':
        # 모델 안에 이미 멀티 지피유 사용함.
        model = build_deepspeech2(
            input_size    = input_size,
            num_classes   = len(vocab),
            rnn_type      = config.rnn_type,
            num_rnn_layers= config.num_encoder_layers,
            rnn_hidden_dim= config.hidden_dim,
            dropout_p     = config.dropout,
            bidirectional = config.use_bidirectional,
            activation    = config.activation,
            device        = device,
        )

    if config.architecture == 'conformer':
            model = Conformer(
                    num_classes=len(vocab), 
                    input_dim=config.n_mels, 
                    # encoder_dim=512, 
                    num_encoder_layers=6,
                    feed_forward_expansion_factor = 2,

            ).to(device)

    return model



if __name__ == '__main__':

    args = argparse.ArgumentParser()

    # DONOTCHANGE
    args.add_argument('--mode', type=str, default='train')   # mode - related to nova system. barely change
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--pause', type=int, default=0)
    # Parameters 
    args.add_argument('--version', type=str, default='train')
    args.add_argument('--make_bow', type=bool, default=True)

    args.add_argument('--use_cuda', type=bool, default=True)
    args.add_argument('--seed', type=int, default=777)
    args.add_argument('--num_epochs', type=int, default=5)
    args.add_argument('--batch_size', type=int, default=16)

    args.add_argument('--save_result_every', type=int, default=2) # 2 
    args.add_argument('--checkpoint_every', type=int, default=1)  # save model every 
    args.add_argument('--print_every', type=int, default=50)
    
    args.add_argument('--dataset', type=str, default='kspon')            # check
    args.add_argument('--output_unit', type=str, default='character')    # check

    # Data Processing
    args.add_argument('--audio_extension', type=str, default='wav')
    args.add_argument('--transform_method', type=str, default='fbank')
    args.add_argument('--feature_extract_by', type=str, default='kaldi')
    
    args.add_argument("--del_silence", type=bool, default=True)
    args.add_argument("--remove_noise", type=bool, default=True)
    args.add_argument("--audio_threshold", type=float, default=0.0075) # 에선 대회 3에서는 0.0885
    args.add_argument("--min_silence_len", type=float, default=3)
    args.add_argument("--make_silence_len", type=float, default=1)
    #MFCC hardCoded
    args.add_argument("--mfcc_max_len", default=1600)

    args.add_argument('--sample_rate', type=int, default=16000)
    args.add_argument('--frame_length', type=int, default=20)
    args.add_argument('--frame_shift', type=int, default=10)
    args.add_argument('--n_mels', type=int, default=80)
    args.add_argument('--freq_mask_para', type=int, default=18)
    args.add_argument('--time_mask_num', type=int, default=4)
    args.add_argument('--freq_mask_num', type=int, default=2)
    args.add_argument('--normalize', type=bool, default=True)
    args.add_argument('--spec_augment', type=bool, default=True)
    args.add_argument('--input_reverse', type=bool, default=False)

    # DataLoader    
    args.add_argument('--num_workers', type=int, default=16)
    args.add_argument('--num_threads', type=int, default=16)
    
    # optimizer
    args.add_argument('--optimizer', type=str, default='adam')      

    # Optimizer lr scheduler options
    args.add_argument("--constant_lr", default=5e-5) #default=5e-5)  # when this is False:   아래의 옵션들이 실행됨.
    args.add_argument('--use_lr_scheduler', type=bool, default=False) ## 
    args.add_argument('--lr_scheduler', type=str, default='tri_stage_lr_scheduler')
    args.add_argument('--init_lr', type=float, default=1e-06)
    args.add_argument('--final_lr', type=float, default=1e-06)
    args.add_argument('--peak_lr', type=float, default=1e-04)
    args.add_argument('--init_lr_scale', type=float, default=1e-02)
    args.add_argument('--final_lr_scale', type=float, default=5e-02)
    args.add_argument('--warmup_steps', type=int, default=1000)
    # args.add_argument('--weight_decay', default=1e-05)
    args.add_argument('--weight_decay', default=False)

    # explode 예방
    args.add_argument('--max_grad_norm', type=int, default=400) ######################## 수정이 필요함.

    # check
    args.add_argument('--reduction', type=str, default='mean')        # check
    args.add_argument('--total_steps', type=int, default=200000)


    args.add_argument('--architecture', type=str, default='conformer') # check
    args.add_argument('--dropout', type=float, default=3e-01)
    args.add_argument('--num_encoder_layers', type=int, default=3)
    args.add_argument('--hidden_dim', type=int, default=1024)
    args.add_argument('--rnn_type', type=str, default='gru')           # check
    args.add_argument('--max_len', type=int, default=400)              #############################이부분도 ???
    args.add_argument('--activation', type=str, default='hardtanh')
    args.add_argument('--use_bidirectional', type=bool, default=True) # maybe decoder aggregator

    args.add_argument('--teacher_forcing_ratio', type=float, default=1.0)  # check
    args.add_argument('--teacher_forcing_step', type=float, default=0.0)   # check
    args.add_argument('--min_teacher_forcing_ratio', type=float, default=1.0)  # check
    args.add_argument('--joint_ctc_attention', type=bool, default=False)   # check
    config = args.parse_args()

    if config.version == 'POC':
        config.num_epochs = 1

    warnings.filterwarnings('ignore')

    # seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    # gpu
    device = 'cuda' if config.use_cuda == True else 'cpu'
    
    # thread
    if hasattr(config, "num_threads") and int(config.num_threads) > 0:
        torch.set_num_threads(config.num_threads)

    # vocab 사전 가지고와 / 사전 csv 파일 열어서, pad, sos, eos, blank_id, unk_id 바꿔.
    vocab = load_vocab(
        config,
        sos_id = '<s>',
        eos_id = '</s>',
        pad_id = '[pad]',
        blank_id = '<blank>',
        # unk_id = '[unk]'
    )

    print('load vocab success')
    print('-'*100)
    print('building model')

    model = build_model(
        input_size=config.n_mels,
        config=config, 
        vocab=vocab, 
        device=device)

    print(model)
    print('build model success')
    print("-"*100)

    # load optimizer
    optimizer = get_optimizer(model, config)
    # bind_model(model, optimizer=optimizer)
    bind_model(model, config)

    cer_metric = get_metric(metric_name='CER', vocab=vocab) #####################  다른 평가 지표 추가해야함.
    wer_metric = get_metric(metric_name='WER', vocab=vocab) # no working

    if config.pause:
        nova.paused(scope=locals())

    if config.mode == 'train':
        config.dataset_path = os.path.join(DATASET_PATH, 'train', 'train_data') # 트레이닝 데이터 셋 위치 
        label_path = os.path.join(DATASET_PATH, 'train', 'train_label')         # 정답지 위치 : train_label이 csv인가 봅니다. -> columns : audio_path, transcript(sentence)
        preprocessing(label_path, os.getcwd()) # current_path/label.csv 가 존재합니다. (git에도 제가 일부를 올렸음.) # transcript

        # config.version == 'POC' 일 경우, 일부 데이터만 갖고 train_dataset, valid_datset 구성됨.
        train_dataset, valid_dataset = split_dataset(
            config, 
            os.path.join(os.getcwd(), 'transcripts.txt'), 
            vocab
        ) # data 부분에서 무음 처리 하는 부분과 broadcasting 부분 바꿔야함.
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=CustomPadFill(0,config),            # 배치 내에서 max값에 padding을 해줬는데, for 사용함 : 속도 느림. torch.pad(?) 사용하자. 적극적으로 broadcasting사용 -> rainism repository : asfl kaggle : 참고
            num_workers=config.num_workers,
            drop_last=True
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=CustomPadFill(0, config),
            num_workers=config.num_workers,
            drop_last=True
        )

        print("#"*100)
        _seqs, _targets, _seq_lengths, _target_lengths = next(iter(train_loader))
        print(f"TRAINING data shape")
        print(f'TRAINING DATA : seqs : {_seqs.shape}')
        print(f'TRAINING DATA : targets : {_targets.shape}')
        print(f'TRAINING DATA : seq lengths : {_seq_lengths}')

        # print(f'TRAINING DATA : target lengths : {_target_lengths.shape}')
        print("#"*100)
        print(f"number of dataset : {len(train_loader)}")
        print(_seqs)
        print(_targets)
        print(_seq_lengths)
        print(_target_lengths)


        # lr 스케쥴 적용한 것과 아닌것.
        if config.use_lr_scheduler:
            lr_scheduler = get_lr_scheduler(config, optimizer, len(train_dataset)) # learning scheduler 적용했네.
            optimizer = Optimizer(optimizer, lr_scheduler, int(len(train_dataset)*config.num_epochs), config.max_grad_norm)

        criterion = get_criterion(config, vocab) # CTC loss


        train_begin_time = time.time()


        for epoch in range(config.num_epochs):
            print('[INFO] Epoch %d start' % epoch)

            # train

            model, train_loss, train_cer = training(
                config,
                train_loader,
                optimizer,
                model,
                criterion,
                cer_metric,
                wer_metric,
                train_begin_time,
                device
            )

            print('[INFO] Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

            valid_begin_time = time.time()

            # valid
            _, valid_loss, valid_cer = validating(
                config,
                valid_loader,
                optimizer,
                model,
                criterion,
                cer_metric,
                wer_metric,
                train_begin_time,
                device,
                vocab=vocab
            )

            print('[INFO] Epoch %d (Validation) Loss %0.4f  CER %0.4f' % (epoch, valid_loss, valid_cer))

            nova.report(
                summary=True,
                epoch=epoch,
                train_loss=train_loss,
                train_cer=train_cer,
                val_loss=valid_loss,
                val_cer=valid_cer
            )

            dict_for_infer = {
                'model' : model.state_dict(),
                'config' : config,
                'vocab' : vocab,
            }

            if epoch % config.checkpoint_every == 0:
                nova.save(epoch)

            torch.cuda.empty_cache()
            print(f'[INFO] epoch {epoch} is done')
        print('[INFO] train process is done')