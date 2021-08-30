import torch
import matplotlib.pyplot as plt
import torchaudio
import math
import cdpam
import semetrics
import norbert
import scipy
from scipy.io.wavfile import write
from sys import platform
if platform == "linux":
    from pypesq import pesq
from pystoi import stoi
from torchaudio.backend.sox_io_backend import load
from config import VOICEBANK_ROOT, VOICEBANK_ROOT, OUTPUT_FILES, MATLAB_ROOT, Config, hparams
from network_functions import *
from complexPyTorch.complexFunctions import complex_matmul
from data import *
from torch.utils.data import DataLoader
from tqdm import tqdm

config = Config()
partition = preprocess(config)

# torch.cuda.empty_cache()


RECEPTIVE_FEILD = 0

# Key is the name of the layer and the value is the
# array consisting of kernel size, stride and padding respectively
# alex_net = {
#     'conv1': [11, 4, 0],
#     'pool1': [3, 2, 0],
#     'conv2': [5, 1, 2],
#     'pool2': [3, 2, 0],
#     'conv3': [3, 1, 1],
#     'conv4': [3, 1, 1],
#     'conv5': [3, 1, 1],
#     'pool5': [3, 2, 0],
#     'fc6-conv': [6, 1, 0],
#     'fc7-conv': [1, 1, 0]
# }
# rv = {
#     'conv1': [7, 2, math.floor(7/2)],
#     'conv2': [7, 2, math.floor(7/2)],
#     'conv3': [5, 2, math.floor(5/2)],
#     'conv4': [5, 2, math.floor(5/2)],
#     'conv5': [3, 2, math.floor(3/2)],
#     'conv6': [3, 2, math.floor(3/2)],
#     'conv7': [3, 2, math.floor(3/2)]
# }

# class ReceptiveFieldCalculator():
#     def calculate(self, architecture, input_image_size):
#         input_layer = ('input_layer', input_image_size, 1, 1, 0.5)
#         self._print_layer_info(input_layer)
        
#         for key in architecture:
#             current_layer = self._calculate_layer_info(architecture[key], input_layer, key)
#             self._print_layer_info(current_layer)
#             input_layer = current_layer
            
#     def _print_layer_info(self, layer):
#         print(f'------')
#         print(f'{layer[0]}: n = {layer[1]}; r = {layer[2]}; j = {layer[3]}; start = {layer[4]}')     
#         print(f'------')
            
#     def _calculate_layer_info(self, current_layer, input_layer, layer_name):
#         n_in = input_layer[1]
#         j_in = input_layer[2]
#         r_in = input_layer[3]
#         start_in = input_layer[4]
        
#         k = current_layer[0]
#         s = current_layer[1]
#         p = current_layer[2]

#         n_out = math.floor((n_in - k + 2*p)/s) + 1
#         padding = (n_out-1)*s - n_in + k 
#         p_right = math.ceil(padding/2)
#         p_left = math.floor(padding/2)

#         j_out = j_in * s
#         r_out = r_in + (k - 1)*j_in
#         start_out = start_in + ((k-1)/2 - p_left)*j_in
#         return layer_name, n_out, j_out, r_out, start_out

# calculator = ReceptiveFieldCalculator()
# calculator.calculate(rv, 256)


SCALING = 0

# orig = torch.FloatTensor(400, 400).uniform_(config.data_minR, config.data_maxR)
# # orig_view_real = torch.view_as_real(orig) 
# print("orig min and max: ", torch.min(orig), torch.max(orig))

# scaled = (orig - config.data_minR) / (config.data_maxR - config.data_minR)
# print("scaled min and max: ", torch.min(scaled), torch.max(scaled))

# scaled_back = ((config.data_maxR + config.data_minR) * ((scaled - 0) / (1 - 0))) + config.data_minR
# print("scaled back min and max: ", torch.min(scaled_back), torch.max(scaled_back))


DATA_LOADING = 0

# orig, sr = load(VOICEBANK_ROOT + "clean_testset_wav/" + "p232_010.wav")
# write(OUTPUT_FILES + "voltest.wav", config.file_sr, orig[0].numpy())

# orig, sr = load(VOICEBANK_ROOT + "clean_testset_wav/" + "p232_010.wav", normalize=True)
# resampler = torchaudio.transforms.Resample(sr, config.sr, dtype=orig.dtype)
# resampled = resampler(orig) 
# write(OUTPUT_FILES + "orig.wav", config.file_sr, orig[0].numpy())
# write(OUTPUT_FILES + "resampled.wav", config.sr, resampled[0].numpy())

# clean_audio, _ = load(VOICEBANK_ROOT + "clean_testset_wav/" + "p232_053.wav", normalize=True)
# clean_audio = torch.squeeze(clean_audio)
# if clean_audio.shape[0] % config.hop_length != 0:
#         remainder = clean_audio.shape[0] % config.hop_length
#         new_length = clean_audio.shape[0] - remainder
#         clean_audio = clean_audio[:new_length]
# else:
#         new_length = clean_audio.shape[0]

# noisy_audio, _ = load(VOICEBANK_ROOT + "noisy_testset_wav/" + "p232_053.wav", normalize=True)
# noisy_audio = torch.squeeze(noisy_audio)[:new_length]
# noise_audio = noisy_audio - clean_audio
# print(noise_audio.shape, noisy_audio.shape, clean_audio.shape, "\n")

# clean_data = torch.stft(clean_audio, n_fft=config.fft_size, hop_length=config.hop_length, \
#         win_length=config.window_length, window=config.window, return_complex=True, \
#         normalized=config.normalise_stft)
# noisy_data = torch.stft(noisy_audio, n_fft=config.fft_size, hop_length=config.hop_length, \
#         win_length=config.window_length, window=config.window, return_complex=True, \
#         normalized=config.normalise_stft)
# noise_data = torch.stft(noise_audio, n_fft=config.fft_size, hop_length=config.hop_length, \
#         win_length=config.window_length, window=config.window, return_complex=True, \
#         normalized=config.normalise_stft) 

# print("clean_data mag: ", torch.min(torch.abs(clean_data)), torch.max(torch.abs(clean_data)))
# print("clean_data phase: ", torch.min(torch.angle(clean_data)), torch.max(torch.angle(clean_data)))
# print("clean_data real: ", torch.min(clean_data.real), torch.max(clean_data.real))
# print("clean_data imag: ", torch.min(clean_data.imag), torch.max(clean_data.imag))
# clean_data_db, ref = amp_tensor_to_dB_tensor(clean_data)
# print("clean_data_db: ", torch.min(torch.abs(clean_data_db)), torch.max(torch.abs(clean_data_db)))
# clean_data_recon = dB_tensor_to_amp_tensor(clean_data_db, ref)
# print("clean_data_recon: ", torch.min(torch.abs(clean_data_recon)), torch.max(torch.abs(clean_data_recon)))

# audio_recon = torch.istft(clean_data, n_fft=config.fft_size, hop_length=config.hop_length, \
#         win_length=config.window_length, normalized=config.normalise_stft)
# write(OUTPUT_FILES + "audio_recon.wav", config.sr, audio_recon.cpu().numpy())

# clean_mag = torch.abs(clean_data)
# noisy_mag = torch.abs(noisy_data)
# noise_mag = torch.abs(noise_data)
# clean_phase = torch.angle(clean_data)
# noisy_phase = torch.angle(noisy_data)
# noise_phase = torch.angle(noise_data)


# PESQ_TESTING = 0 #https://www.microtronix.ca/pesq.html
# clean, sr = load("../test_files/Or272.wav", normalize=True)
# clean = torch.squeeze(clean)
# deg1, sr = load("../test_files/Dg001.wav", normalize=True)
# deg1 = torch.squeeze(deg1)
# deg2, sr = load("../test_files/Dg002.wav", normalize=True)
# deg2 = torch.squeeze(deg2)

# min_len = min(clean.shape[0], deg1.shape[0], deg2.shape[0])
# clean = clean[:min_len]
# deg1 = deg1[:min_len]
# deg2 = deg2[:min_len]

# pesq1 = pesq(clean.cpu().numpy(), deg1.cpu().numpy(), sr)
# pesq2 = pesq(clean.cpu().numpy(), deg2.cpu().numpy(), sr)

# print("pesq1: ", pesq1)
# print("pesq2: ", pesq2)


WIENER_AND_SPECSUB = 0

# pesq_noisy_arr = []
# pesq_wiener_arr = []
# stoi_noisy_arr = []
# stoi_wiener_arr = []
# for step, ID in enumerate(tqdm(partition["test"])):
#         noisy_audio, _ = load(VOICEBANK_ROOT + "noisy_testset_wav/" + ID + ".wav", normalize=config.normalise_audio)
#         clean_audio, _ = load(VOICEBANK_ROOT + "clean_testset_wav/" + ID + ".wav", normalize=config.normalise_audio)
#         noisy_audio = torch.squeeze(noisy_audio).float()
#         clean_audio = torch.squeeze(clean_audio).float()
                                                            
#         noisy_data = torch.stft(noisy_audio,
#                 n_fft=config.fft_size,
#                 hop_length=config.hop_length,
#                 win_length=config.window_length,
#                 window=config.window,
#                 return_complex=True,
#                 normalized=config.normalise_stft)

#         clean_data = torch.stft(clean_audio,
#                 n_fft=config.fft_size,
#                 hop_length=config.hop_length,
#                 win_length=config.window_length,
#                 window=config.window,
#                 return_complex=True,
#                 normalized=config.normalise_stft)
#         clean_mag = torch.abs(clean_data)

#         clean_mag = torch.unsqueeze(torch.unsqueeze(clean_mag, 2), 2)
#         noisy_data = torch.unsqueeze(noisy_data, 2)

#         wiener_est = torch.tensor(norbert.wiener(clean_mag.numpy(), noisy_data.numpy(), iterations=1, use_softmask=True, eps=10e-8))
#         wiener_est = torch.squeeze(torch.squeeze(wiener_est))

#         est_audio = torch.istft(wiener_est, n_fft=config.fft_size, hop_length=config.hop_length, \
#                         win_length=config.window_length, normalized=config.normalise_stft)
#         min_length = min(noisy_audio.shape[0], clean_audio.shape[0], est_audio.shape[0])
#         noisy_audio = noisy_audio[:min_length]
#         clean_audio = clean_audio[:min_length]
#         est_audio = est_audio[:min_length]

#         pesq_noisy = pesq(clean_audio.cpu().numpy(), noisy_audio.cpu().numpy(), config.sr)
#         pesq_wiener = pesq(clean_audio.cpu().numpy(), est_audio.cpu().numpy(), config.sr)
#         stoi_noisy = stoi(clean_audio.cpu().numpy(), noisy_audio.cpu().numpy(), config.sr)
#         stoi_wiener = stoi(clean_audio.cpu().numpy(), est_audio.cpu().numpy(), config.sr)

#         if not isnan(pesq_noisy):
#             pesq_noisy_arr.append(pesq_noisy)

#         if not isnan(pesq_wiener):
#             pesq_wiener_arr.append(pesq_wiener)

#         if not isnan(stoi_noisy):
#             stoi_noisy_arr.append(stoi_noisy)

#         if not isnan(stoi_wiener): 
#             stoi_wiener_arr.append(stoi_wiener)

# pesq_noisy_avg = sum(pesq_noisy_arr) / len(pesq_noisy_arr)
# pesq_wiener_avg = sum(pesq_wiener_arr) / len(pesq_wiener_arr)
# stoi_noisy_avg = sum(stoi_noisy_arr) / len(stoi_noisy_arr)
# stoi_wiener_avg = sum(stoi_wiener_arr) / len(stoi_wiener_arr)

# print("pesq_noisy avg: ", pesq_noisy_avg)
# print("pesq_wiener avg: ", pesq_wiener_avg)
# print("stoi_noisy avg: ", stoi_noisy_avg)
# print("stoi_wiener avg: ", stoi_wiener_avg)

# pesq_ss_arr = []
# stoi_ss_arr = []
# for step, ID in enumerate(tqdm(partition["test"])):
#         ss_audio, _ = load("/homes/jhw31/Documents/Project/" + "spec_sub_test/" + ID + ".wav", \
#                 normalize=config.normalise_audio)
#         clean_audio, _ = load(VOICEBANK_ROOT + "clean_testset_wav/" + ID + ".wav", \
#                 normalize=config.normalise_audio)
#         ss_audio = torch.squeeze(ss_audio)
#         clean_audio = torch.squeeze(clean_audio)

#         min_length = min(ss_audio.shape[0], clean_audio.shape[0])
#         ss_audio = ss_audio[:min_length]
#         clean_audio = clean_audio[:min_length]

#         pesq_ss = pesq(clean_audio.cpu().numpy(), ss_audio.cpu().numpy(), config.sr)
#         stoi_ss = stoi(clean_audio.cpu().numpy(), ss_audio.cpu().numpy(), config.sr)

#         pesq_ss_arr.append(pesq_ss)
#         stoi_ss_arr.append(stoi_ss)

# pesq_ss_avg = sum(pesq_ss_arr) / len(pesq_ss_arr)
# stoi_ss_avg = sum(stoi_ss_arr) / len(stoi_ss_arr)

# print("pesq_ss avg: ", pesq_ss_avg)
# print("stoi_ss avg: ", stoi_ss_avg)


LOSS_TESTING = 0

# print("cL1: ", cL1)
# caudio = torch.istft(ccomp, n_fft=config.fft_size, hop_length=config.hop_length, \
#         win_length=config.window_length, normalized=config.normalise_stft)
# write(OUTPUT_FILES + "caudio.wav", config.sr, caudio.numpy())
# cSiSNR = config.SiSNR(noise_audio, caudio)
# print("cSiSNR: ", cSiSNR)
# cwSDR = config.wSDR(torch.unsqueeze(noisy_audio, 0), torch.unsqueeze(noise_audio, 0), torch.unsqueeze(caudio, 0))
# print("cwSDR: ", cwSDR)
# cSTOI = stoi(noise_audio.numpy(), caudio.numpy(), config.sr)
# print("cSTOI: ", cSTOI)

# clean_data_mod = torch.complex(clean_data[...,0], noisy_data[...,1])

# mod_clean_data = clean_data
# print(clean_data[...,1])

# L1_loss = config.noise_criterion(noisy_data, clean_data)
# print("L1 loss: ", L1_loss) 

# altered = complex_mat_mult(noisy_data, mask)

# mod_audio = torch.istft(mod_clean_data, n_fft=config.fft_size, hop_length=config.hop_length, \
#         win_length=config.window_length, normalized=config.normalise_stft) 

# write(OUTPUT_FILES + "orig.wav", config.sr, clean_audio.numpy())
# write(OUTPUT_FILES + "mod.wav", config.file_sr, mod_audio.numpy())

# L1 = config.noise_criterion(torch.abs(clean_data), torch.abs(noisy_data))
# print("L1: ", L1)

# SiSNR = config.speech_criterion(clean_audio, noisy_audio)
# print("SiSNR: ", SiSNR)

# pesq_i = pesq(clean_data.cpu().numpy(), noisy_data.cpu().numpy(), 16000)
# print(pesq_i)


MATLAB_TESTING = 0

# clean_audio, _ = load(MATLAB_ROOT + "audio/" + "clean.wav", normalize=True)
# clean_audio = torch.squeeze(clean_audio)
# if clean_audio.shape[0] % config.hop_length != 0:
#         remainder = clean_audio.shape[0] % config.hop_length
#         new_length = clean_audio.shape[0] - remainder
#         clean_audio = clean_audio[:50688]
# else:
#         new_length = clean_audio.shape[0]

# noisy_audio, _ = load(MATLAB_ROOT + "audio/" + "noisy.wav", normalize=True)
# noisy_audio = torch.squeeze(noisy_audio)[:50688]
# noise_audio = noisy_audio - clean_audio

# MATLAB_enhanced_audio, _ = load(MATLAB_ROOT + "audio/" + "enhanced.wav", normalize=True)
# MATLAB_enhanced_audio = torch.squeeze(MATLAB_enhanced_audio)[:50688]

# print(noise_audio.shape[0], noisy_audio.shape[0], clean_audio.shape[0], MATLAB_enhanced_audio.shape[0], "\n")

# pesq_noisy = pesq(clean_audio.cpu().numpy(), noisy_audio.cpu().numpy(), config.sr)
# pesq_MATLAB_enhanced = pesq(clean_audio.cpu().numpy(), MATLAB_enhanced_audio.cpu().numpy(), config.sr)

# print("pesq_noisy: ", pesq_noisy)
# print("pesq_MATLAB_enhanced: ", pesq_MATLAB_enhanced)

# stoi_noisy = stoi(clean_audio.cpu().numpy(), noisy_audio.cpu().numpy(), config.sr)
# stoi_MATLAB_enhanced = stoi(clean_audio.cpu().numpy(), MATLAB_enhanced_audio.cpu().numpy(), config.sr)

# print("stoi_noisy: ", stoi_noisy)
# print("stoi_MATLAB_enhanced: ", stoi_MATLAB_enhanced)

# train_set = VoiceBankDataset(partition['train'], config, mode="train", seed=config.seed)
# train_loader = DataLoader(train_set, **config.data_params)

# validation_set = VoiceBankDataset(partition['val'], config, mode="val", seed=config.seed)
# validation_loader = DataLoader(validation_set, **config.data_params)


RESHAPING = 0
# e7 = torch.rand(32, 256, 2, 16)
# latent_shape = e7.shape
# flattened = torch.flatten(e7, 2, 3).permute(0, 2, 1)
# unflattened = flattened.permute(0, 2, 1).reshape(latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3])
# print("are tensors equal? ", torch.equal(e7, unflattened))


MIN_AND_MAX_OF_DATASET = 0

# maximum = 0
# minimum = 0
# for step, ID in enumerate(tqdm(partition["train"])):
#         clean_data, _ = load(VOICEBANK_ROOT + "clean_trainset_wav/" + ID + ".wav", \
#                 normalize=config.normalise_audio)
#         noisy_data, _ = load(VOICEBANK_ROOT + "clean_trainset_wav/" + ID + ".wav", \
#                 normalize=config.normalise_audio)
#         clean_data = torch.squeeze(clean_data).float()
#         noisy_data = torch.squeeze(noisy_data).float()
#         noise_data = noisy_data - clean_data

#         noise_data = torch.stft(noise_data,
#                 n_fft=config.fft_size,
#                 hop_length=config.hop_length,
#                 win_length=config.window_length,
#                 window=config.window,
#                 return_complex=True,
#                 normalized=config.normalise_stft)[1:int(config.fft_size / 2) + 1, :]
                                                            
#         noisy_data = torch.stft(noisy_data,
#                 n_fft=config.fft_size,
#                 hop_length=config.hop_length,
#                 win_length=config.window_length,
#                 window=config.window,
#                 return_complex=True,
#                 normalized=config.normalise_stft)[1:int(config.fft_size / 2) + 1, :]

#         clean_data = torch.stft(clean_data,
#                 n_fft=config.fft_size,
#                 hop_length=config.hop_length,
#                 win_length=config.window_length,
#                 window=config.window,
#                 return_complex=True,
#                 normalized=config.normalise_stft)[1:int(config.fft_size / 2) + 1, :]

#         # noise_max = torch.max(torch.view_as_real(noise_data))
#         # noisy_max = torch.max(torch.view_as_real(noisy_data))
#         # clean_max = torch.max(torch.view_as_real(clean_data))

#         # noise_min = torch.min(torch.view_as_real(noise_data))
#         # noisy_min = torch.min(torch.view_as_real(noisy_data))
#         # clean_min = torch.min(torch.view_as_real(clean_data))

#         noise_max = torch.max(torch.abs(noise_data))
#         noisy_max = torch.max(torch.abs(noisy_data))
#         clean_max = torch.max(torch.abs(clean_data))

#         noise_min = torch.min(torch.abs(noise_data))
#         noisy_min = torch.min(torch.abs(noisy_data))
#         clean_min = torch.min(torch.abs(clean_data))

#         max_max = max(noise_max, noisy_max, clean_max)
#         min_min = min(noise_min, noisy_min, clean_min)

#         if max_max > maximum:
#                 maximum = max_max

#         if min_min < minimum:
#                 minimum = min_min

# for step, ID in enumerate(tqdm(partition["val"])):
#         clean_data, _ = load(VOICEBANK_ROOT + "clean_trainset_wav/" + ID + ".wav", \
#                 normalize=config.normalise_audio)
#         noisy_data, _ = load(VOICEBANK_ROOT + "clean_trainset_wav/" + ID + ".wav", \
#                 normalize=config.normalise_audio)
#         clean_data = torch.squeeze(clean_data).float()
#         noisy_data = torch.squeeze(noisy_data).float()
#         noise_data = noisy_data - clean_data

#         noise_data = torch.stft(noise_data,
#                 n_fft=config.fft_size,
#                 hop_length=config.hop_length,
#                 win_length=config.window_length,
#                 window=config.window,
#                 return_complex=True,
#                 normalized=config.normalise_stft)[1:int(config.fft_size / 2) + 1, :]
                                                            
#         noisy_data = torch.stft(noisy_data,
#                 n_fft=config.fft_size,
#                 hop_length=config.hop_length,
#                 win_length=config.window_length,
#                 window=config.window,
#                 return_complex=True,
#                 normalized=config.normalise_stft)[1:int(config.fft_size / 2) + 1, :]

#         clean_data = torch.stft(clean_data,
#                 n_fft=config.fft_size,
#                 hop_length=config.hop_length,
#                 win_length=config.window_length,
#                 window=config.window,
#                 return_complex=True,
#                 normalized=config.normalise_stft)[1:int(config.fft_size / 2) + 1, :]

#         # noise_max = torch.max(torch.view_as_real(noise_data))
#         # noisy_max = torch.max(torch.view_as_real(noisy_data))
#         # clean_max = torch.max(torch.view_as_real(clean_data))

#         # noise_min = torch.min(torch.view_as_real(noise_data))
#         # noisy_min = torch.min(torch.view_as_real(noisy_data))
#         # clean_min = torch.min(torch.view_as_real(clean_data))

#         noise_max = torch.max(torch.abs(noise_data))
#         noisy_max = torch.max(torch.abs(noisy_data))
#         clean_max = torch.max(torch.abs(clean_data))

#         noise_min = torch.min(torch.abs(noise_data))
#         noisy_min = torch.min(torch.abs(noisy_data))
#         clean_min = torch.min(torch.abs(clean_data))

#         max_max = max(noise_max, noisy_max, clean_max)
#         min_min = min(noise_min, noisy_min, clean_min)

#         if max_max > maximum:
#                 maximum = max_max

#         if min_min < minimum:
#                 minimum = min_min

# print("max: ", maximum)
# print("min: ", minimum)
