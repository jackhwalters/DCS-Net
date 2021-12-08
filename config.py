import os
import torchaudio
from sys import platform
from torch import nn, hann_window, cuda, optim
from complexPyTorch.complexFunctions import complex_relu
from network_functions import SiSNR, wSDR, complex_lrelu

if platform == "linux":
    VOICEBANK_ROOT = "/homes/jhw31/Documents/Project/DS_10283_1942/"
    torchaudio.set_audio_backend("sox_io")
elif platform == "darwin":
    VOICEBANK_ROOT = "/Volumes/Work/Project/Datasets/DS_10283_1942/"
    torchaudio.set_audio_backend("sox_io")
PROJECT_ROOT = "./"
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "model_output/")
DATA_JSON = os.path.join(PROJECT_ROOT, "data_json/")
OUTPUT_FILES = os.path.join(PROJECT_ROOT, "output_files/")
MATLAB_ROOT = "../MATLAB/"

# noise_loss_type
# 0: L1
# 1: wSDR
# 2: L1L1
# 3: wSDRL1(wave)
# 4: wSDRL1(spec)
# 5: wSDRMSE(spec)

# speech_loss_type
# 0 SiSNR
hparams = {'lr': 10e-5,
            'noise_alpha': 1,
            'speech_alpha': 0.25,
            'no_of_layers': 7,
            'channels': [1, 16, 32, 64, 128, 256, 256, 256],
            'lstm_layers': 2,
            'lstm_bidir': True,
            'noise_loss_type': 4,
            'speech_loss_type': 0,
            'skip_concat': True,
            'dropout': True,
            'dropout_conv': 0.1,
            'dropout_fc': 0.2,
            'batch_size': 32,
            'optim_eps': 10e-7,
            'optim_weight_decay': 10e-5,
            'optim_amsgrad': True,
            'gradient_clip_val': 10.0,
            'gradient_clip_algorithm': "norm",
            'stochastic_weight_avg': True}

class Config(object):
    def __init__(self):
        self.tune = False
        self.load_data_into_RAM = False
        self.sr = 16000
        self.file_sr = 48000
        self.resample = torchaudio.transforms.Resample(orig_freq=self.file_sr, new_freq=self.sr)
        self.train_val_split = 80
        self.max_epochs = 200
        self.num_loader_workers = cuda.device_count() * 4 if cuda.is_available() else 0
        self.num_gpus = cuda.device_count() if cuda.is_available() else 0
        self.data_params = {'batch_size': hparams['batch_size'],
                    'shuffle': True,
                    'num_workers': self.num_loader_workers,
                    'pin_memory': True}
        self.precision = 32
        
        self.fft_size = 512
        self.window_length = self.fft_size
        self.hop_length = 64
        self.window = hann_window(window_length=self.window_length) 
        self.normalise_audio = True
        self.normalise_stft = False

        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.SiSNR = SiSNR()
        self.wSDR = wSDR() 
        self.kernel_sizeE = [7,7,5,5,3,3,3]
        self.kernel_sizeD = [3,3,3,3,3,3,3]
        self.paddingE = [self.kernel_sizeE[0] // 2,
                        self.kernel_sizeE[1] // 2,
                        self.kernel_sizeE[2] // 2,
                        self.kernel_sizeE[3] // 2,
                        self.kernel_sizeE[4] // 2,
                        self.kernel_sizeE[5] // 2,
                        self.kernel_sizeE[6] // 2]
        self.paddingD = [self.kernel_sizeD[0] // 2,
                        self.kernel_sizeD[1] // 2,
                        self.kernel_sizeD[2] // 2,
                        self.kernel_sizeD[3] // 2,
                        self.kernel_sizeD[4] // 2,
                        self.kernel_sizeD[5] // 2,
                        self.kernel_sizeD[6] // 2]
        self.strideE = [(2,2),(2,2),(2,2),(2,1),(2,1),(2,1),(2,1)]
        self.strideD = (1,1)
        self.initialisation_distribution = nn.init.xavier_uniform_
        self.RactivationE = nn.functional.relu
        self.RactivationD = nn.functional.leaky_relu
        self.CactivationE = complex_relu 
        self.CactivationD = complex_lrelu
        self.upsample_scale_factor = [(2,1), (2,1), (2,1), (2,1), (2,2), (2,2), (2,2)]
        self.upsampling_mode = 'nearest'
        
        self.receptive_field_freq = 291 * (self.sr / self.fft_size)
        self.receptive_field_time = 291 / (self.sr / self.hop_length)
        self.integer_win_size = int(((1000 / (self.sr / self.hop_length)) \
                                * (self.window_length / 2) / 1000) * self.sr)

        self.val_log_sample_size = 1
        self.seed = 0
        self.data_minC = -69.4638
        self.data_maxC = 69.9589
        self.data_minR = 0
        self.data_maxR = 74.6140
