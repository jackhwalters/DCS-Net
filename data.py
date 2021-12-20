import os
import shutil
import re
import json
import torch
import numpy as np
import pytorch_lightning as pl
from network_functions import check_inf_neginf_nan
from sys import platform
from scipy.io.wavfile import write
from math import floor
if platform == "win32":
    from torchaudio.backend.soundfile_backend import load
else:
    from torchaudio.backend.sox_io_backend import load
from tqdm import tqdm
from os import walk
from config import PROJECT_ROOT, VOICEBANK_ROOT, DATA_JSON, OUTPUT_FILES, hparams


class VoiceBankDataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, config, mode, seed):
        pl.seed_everything(seed)
        self.list_IDs = list_IDs
        # self.labels = labels
        self.config = config
        self.mode = mode
        if self.mode == "train" or self.mode == "val":
            if self.config.load_data_into_RAM:
                self.clean_dir = {}
                print("Loading clean trainset into RAM...\n")
                for file in tqdm(self.list_IDs):
                    audio, _ = load(VOICEBANK_ROOT + "clean_trainset_{}spk_wav/".format(hparams['dataset_type']) \
                         + file + ".wav", normalize=self.config.normalise_audio)
                    self.clean_dir[os.path.splitext(file)[-2]] = audio
                self.noisy_dir = {}
                print("Loading noisy trainset into RAM...\n")
                for file in tqdm(self.list_IDs):
                    audio, _ = load(VOICEBANK_ROOT + "noisy_trainset_{}spk_wav/".format(hparams['dataset_type']) \
                        + file + ".wav", normalize=self.config.normalise_audio)
                    self.noisy_dir[os.path.splitext(file)[-2]] = audio
            elif not self.config.load_data_into_RAM:
                print("Reading train and validation from disk...\n")
                self.clean_dir = VOICEBANK_ROOT + "clean_trainset_{}spk_wav/".format(hparams['dataset_type'])
                self.noisy_dir = VOICEBANK_ROOT + "noisy_trainset_{}spk_wav/".format(hparams['dataset_type'])
        elif self.mode == "test":
            if self.config.load_data_into_RAM:
                self.clean_dir = {}
                print("Loading clean testset into RAM...\n")
                for file in tqdm(self.list_IDs):
                    audio, _ = load(VOICEBANK_ROOT + "clean_testset_wav/" + file + ".wav", \
                        normalize=self.config.normalise_audio)
                    self.clean_dir[os.path.splitext(file)[-2]] = audio
                self.noisy_dir = {}
                print("Loading noisy testset into RAM...\n")
                for file in tqdm(self.list_IDs):
                    audio, _ = load(VOICEBANK_ROOT + "noisy_testset_wav/" + file + ".wav", \
                        normalize=self.config.normalise_audio)
                    self.noisy_dir[os.path.splitext(file)[-2]] = audio
            elif not self.config.load_data_into_RAM:
                print("Reading test data from disk...\n")
                self.clean_dir = VOICEBANK_ROOT + "clean_testset_wav/"
                self.noisy_dir = VOICEBANK_ROOT + "noisy_testset_wav/"

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        if self.config.load_data_into_RAM:
            clean_data = self.clean_dir[ID]
            noisy_data = self.noisy_dir[ID]
        elif not self.config.load_data_into_RAM:
            clean_data, _ = load(self.clean_dir + ID + ".wav", \
                normalize=self.config.normalise_audio)
            noisy_data, _ = load(self.noisy_dir + ID + ".wav", \
                normalize=self.config.normalise_audio)

        clean_data = torch.squeeze(clean_data).float()
        noisy_data = torch.squeeze(noisy_data).float()

        # if self.mode == "test":
        clean_data = self.config.resample(clean_data)
        noisy_data = self.config.resample(noisy_data)

        if clean_data.shape[0] != noisy_data.shape[0]:
            raise Exception("clean_data and noisy_data are not the same length")

        data_len = clean_data.shape[0]
        windowed_data_len = self.config.integer_win_size - self.config.hop_length

        # print("data_len: ", data_len)
        # print("windowed_data_len: ", windowed_data_len)
        if windowed_data_len > data_len:
            clean_data = torch.nn.functional.pad(clean_data, (0, windowed_data_len-data_len))
            noisy_data = torch.nn.functional.pad(noisy_data, (0, windowed_data_len-data_len))
            start_point = 0
        else:
            start_window_len = data_len - windowed_data_len
            start_point = torch.randint(0, start_window_len, (1,))

        clean_data = clean_data[start_point:start_point + windowed_data_len]
        noisy_data = noisy_data[start_point:start_point + windowed_data_len]
        noise_data = noisy_data - clean_data

        check_inf_neginf_nan(clean_data, "Found inf, neginf or nan in clean audio!")
        check_inf_neginf_nan(noisy_data, "Found inf, neginf or nan in noisy audio!")
        check_inf_neginf_nan(noise_data, "Found inf, neginf or nan in noise audio!")

        #F x T
        clean_data = torch.stft(clean_data,
                    n_fft=self.config.fft_size,
                    hop_length=self.config.hop_length,
                    win_length=self.config.window_length,
                    window=self.config.window,
                    return_complex=True,
                    normalized=self.config.normalise_stft)[1:int(self.config.fft_size / 2) + 1, :]

        noise_data = torch.stft(noise_data,
                    n_fft=self.config.fft_size,
                    hop_length=self.config.hop_length,
                    win_length=self.config.window_length,
                    window=self.config.window,
                    return_complex=True,
                    normalized=self.config.normalise_stft)[1:int(self.config.fft_size / 2) + 1, :]
                                                            
        noisy_data = torch.stft(noisy_data,
                    n_fft=self.config.fft_size,
                    hop_length=self.config.hop_length,
                    win_length=self.config.window_length,
                    window=self.config.window,
                    return_complex=True,
                    normalized=self.config.normalise_stft)[1:int(self.config.fft_size / 2) + 1, :]

        check_inf_neginf_nan(clean_data, "Found inf, neginf or nan in clean data STFT!")
        check_inf_neginf_nan(noise_data, "Found inf, neginf or nan in noise data STFT!")
        check_inf_neginf_nan(noisy_data, "Found inf, neginf or nan in noisy data STFT!")

        if self.mode == "test":
            return noise_data, noisy_data, clean_data, ID, start_point
        else:
            return noise_data, noisy_data, clean_data, ID


def walk_files(dir_path):
    files = []
    for (dirpath, dirnames, filenames) in walk(dir_path):
        files.extend(filenames)
        break
    return files

def preprocess(config):
    partition = {}

    if not os.path.exists(DATA_JSON + "partition.json"):
        print('VoiceBank data JSON files do not exist. Creating now...\n')
        train_val_set_wav = np.array((walk_files(VOICEBANK_ROOT + "clean_trainset_{}spk_wav/".format(hparams['dataset_type'])))) 
        train_val_set_wav = np.core.defchararray.replace(train_val_set_wav, ".wav", "")
        np.random.shuffle(train_val_set_wav)
        train_val_split_index = round(train_val_set_wav.shape[0] * (config.train_val_split/100))
        train_set_wav = train_val_set_wav[:train_val_split_index]
        val_set_wav = train_val_set_wav[train_val_split_index:]

        testset_wav = np.array(walk_files((VOICEBANK_ROOT + "clean_testset_wav/")))
        testset_wav = np.core.defchararray.replace(testset_wav, ".wav", "")
        np.random.shuffle(testset_wav)

        assert len(train_set_wav) == len(set(train_set_wav)), "Duplicate item in train set"
        assert len(val_set_wav) == len(set(val_set_wav)), "Duplicate item in val set"
        assert len(testset_wav) == len(set(testset_wav)), "Duplicate item in test set"

        assert set(train_set_wav).isdisjoint(set(val_set_wav)), "Train and val sets are not disjoint"
        assert set(train_set_wav).isdisjoint(set(testset_wav)), "Train and test sets are not disjoint"
        assert set(val_set_wav).isdisjoint(set(testset_wav)), "Val and test sets are not disjoint"

        partition = {'train': train_set_wav.tolist(),
                'val': val_set_wav.tolist(),
                'test': testset_wav.tolist()}
        
        with open(DATA_JSON + "partition.json", "w+") as fp:
            json.dump(partition, fp)

    else:
        print('VoiceBank data JSON files already exist\n')
        with open(DATA_JSON + "partition.json") as js:
            partition = json.loads(js.read())

    return partition
