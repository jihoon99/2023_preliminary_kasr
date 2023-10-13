import sys, os
import threading
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import random
import math
import pydub

from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from modules.vocab import Vocabulary
from modules.audio.core import load_audio
from modules.audio.parser import SpectrogramParser


class SpectrogramDataset(Dataset, SpectrogramParser):
    """
    Dataset for feature & transcript matching

    Args:
        audio_paths (list): list of audio path
        transcripts (list): list of transcript
        sos_id (int): identification of <start of sequence>
        eos_id (int): identification of <end of sequence>
        spec_augment (bool): flag indication whether to use spec-augmentation or not (default: True)
        config (DictConfig): set of configurations
        dataset_path (str): path of dataset
    """

    def __init__(
            self,
            audio_paths: list,  # list of audio paths
            transcripts: list,  # list of transcript paths
            sos_id: int,  # identification of start of sequence token
            eos_id: int,  # identification of end of sequence token
            config,  # set of arguments
            spec_augment: bool = False,  # flag indication whether to use spec-augmentation of not
            dataset_path: str = None,  # path of dataset,
            audio_extension: str = 'pcm'  # audio extension
    ) -> None:
        super(SpectrogramDataset, self).__init__(
            feature_extract_by=config.feature_extract_by, 
            sample_rate=config.sample_rate,
            n_mels=config.n_mels, 
            frame_length=config.frame_length, 
            frame_shift=config.frame_shift,
            del_silence=config.del_silence, 
            input_reverse=config.input_reverse,
            normalize=config.normalize, 
            freq_mask_para=config.freq_mask_para,
            time_mask_num=config.time_mask_num, 
            freq_mask_num=config.freq_mask_num,
            sos_id=sos_id, 
            eos_id=eos_id, 
            dataset_path=dataset_path,  # config.dataset_path
            transform_method=config.transform_method,
            audio_extension=audio_extension
        )
        self.audio_paths = list(audio_paths)
        self.transcripts = list(transcripts)

        self.augment_methods = [self.VANILLA] * len(self.audio_paths) # self.VANILA : agument 안함.
        self.dataset_size = len(self.audio_paths) # 데이터 갯수
        self._augment(spec_augment) # 증강방법은 나ㅇ에.
        self.shuffle() # -> 또 섞어? DataLoader에 shuffle 파라미어 있는데,,, 

    def __getitem__(self, idx):
        """ get feature vector & transcript """
        feature = self.parse_audio(os.path.join(self.dataset_path, self.audio_paths[idx]), self.augment_methods[idx]) # 해당하는 오디오를 불러옴.
        # 더 들어가면 modules/audio/core에서 load하는 메서드가 있는데, silence remove 하는 부분의 로직이 빈약함. -> 예선전 알고리즘으로 고도화 가능

        if feature is None:
            return None

        transcript, status = self.parse_transcript(self.transcripts[idx])
        # self.transcripts[idx] : 2345 1353 1 3817 2038  이렇게 생겼음. 토큰 단위로 잘라줘야함.
        # sos, eos 토큰 앞뒤로 붙여줌.

        if status == 'err':
            print(self.transcripts[idx])
            print(idx)
        return feature, transcript # feature : spectogram audio, trasncript : list contain tokens
        # feature  부분 shape 더 살펴봐야함
        # wave2vec2.0 pretrained version 사용해서 고도화 가능
        # 다만 데이터가 pretrain하기 충분한 양인지는 모르겠음.

    def parse_transcript(self, transcript):
        """ Parses transcript """
        tokens = transcript.split(' ')
        transcript = list()

        transcript.append(int(self.sos_id))
        for token in tokens:
            try:
                transcript.append(int(token))
                status='nor'
            except:
                print(tokens)
                status='err'
        transcript.append(int(self.eos_id))

        return transcript, status

    def _augment(self, spec_augment):
        """ Spec Augmentation """
        if spec_augment:
            print("Applying Spec Augmentation...")

            for idx in range(self.dataset_size):
                self.augment_methods.append(self.SPEC_AUGMENT)
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])

    def shuffle(self):
        """ Shuffle dataset """
        tmp = list(zip(self.audio_paths, self.transcripts, self.augment_methods))
        random.shuffle(tmp)
        self.audio_paths, self.transcripts, self.augment_methods = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)
    



def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)


def load_dataset(transcripts_path):
    """
    JH :
    ---------------------------
        transcripts_path : txt같은 파일임.
            audio_path, sentence, encoded_sentence 가 한줄 한줄 들어있음.


        한줄씩 불러와서
            audio_path, korean_transcript, transcript라고  오브젝트 할당함.

            

        return
            audio_paths : [file1, file2, file3, ...]
            transcripts : [
                            [1,2,3,...],
                            [103,1853,20183, ...],
                            [5729,10385,20381, ...]
                        ]
            


    ----------------------------

    Provides dictionary of filename and labels

    Args:
        transcripts_path (str): path of transcripts

    Returns: target_dict
        - **target_dict** (dict): dictionary of filename and labels
    """
    audio_paths = list()
    transcripts = list()

    with open(transcripts_path) as f:
        for idx, line in enumerate(f.readlines()):
            try:
                audio_path, korean_transcript, transcript = line.split('\t')
            except:
                print(line)
            transcript = transcript.replace('\n', '')

            audio_paths.append(audio_path)
            transcripts.append(transcript)

    return audio_paths, transcripts # 이것들 다 list 임.


def split_dataset(config, transcripts_path: str, vocab: Vocabulary, valid_size=.2):
    """
    split into training set and validation set.

    Args:
        opt (ArgumentParser): set of options
        transcripts_path (str): path of  transcripts

    Returns: train_batch_num, train_dataset_list, valid_dataset
        - **train_time_step** (int): number of time step for training
        - **trainset_list** (list): list of training dataset
        - **validset** (data_loader.MelSpectrogramDataset): validation dataset
    """


    print("split dataset start !!")
    trainset_list = list()
    validset_list = list()

    audio_paths, transcripts = load_dataset(transcripts_path) # 리스트를 아웃풋으로 내뱉음.
    # 오디오 위치, 인코딩된 문장

    if config.version == 'PoC':
        audio_paths = audio_paths[:1000]
        transcripts = transcripts[:1000]
        
    train_audio_paths, valid_audio_paths, train_transcripts, valid_transcripts = train_test_split(audio_paths,
                                                                                                  transcripts,
                                                                                                  test_size=valid_size) # sklearn train_test_split임.


    # audio_paths & script_paths shuffled in the same order
    # for seperating train & validation
    tmp = list(zip(train_audio_paths, train_transcripts))
    random.shuffle(tmp)
    train_audio_paths, train_transcripts = zip(*tmp)

    # seperating the train dataset by the number of workers

    train_dataset = SpectrogramDataset(
        train_audio_paths,
        train_transcripts,
        vocab.sos_id, vocab.eos_id,
        config=config,
        spec_augment=config.spec_augment,
        dataset_path=config.dataset_path,
        audio_extension=config.audio_extension,
    )

    valid_dataset = SpectrogramDataset(
        valid_audio_paths,
        valid_transcripts,
        vocab.sos_id, vocab.eos_id,
        config=config,
        spec_augment=config.spec_augment,
        dataset_path=config.dataset_path,
        audio_extension=config.audio_extension,
    )

    return train_dataset, valid_dataset


def collate_fn(batch):
    pad_id = 0
    """ functions that pad to the maximum sequence length """

    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    # sort by sequence length for rnn.pack_padded_sequence()
    try:
        batch = [i for i in batch if i != None]
        batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

        seq_lengths = [len(s[0]) for s in batch]
        target_lengths = [len(s[1]) - 1 for s in batch]

        max_seq_sample = max(batch, key=seq_length_)[0]
        max_target_sample = max(batch, key=target_length_)[1]

        max_seq_size = max_seq_sample.size(0)
        max_target_size = len(max_target_sample)

        feat_size = max_seq_sample.size(1)
        batch_size = len(batch)

        seqs = torch.zeros(batch_size, max_seq_size, feat_size)

        targets = torch.zeros(batch_size, max_target_size).to(torch.long)
        targets.fill_(pad_id)

        for x in range(batch_size):
            sample = batch[x]
            tensor = sample[0]
            target = sample[1]
            seq_length = tensor.size(0)

            seqs[x].narrow(0, 0, seq_length).copy_(tensor)
            targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

        seq_lengths = torch.IntTensor(seq_lengths)
        return seqs, targets, seq_lengths, target_lengths
    except Exception as e:
        print(e)