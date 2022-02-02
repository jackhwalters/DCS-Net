import os
import torchaudio
from sys import platform
from torch import nn, hann_window, cuda, optim
from complexPyTorch.complexLayers import ComplexReLU
from network_functions import SiSNR, wSDR, ComplexLReLU

if platform == "linux":
    VOICEBANK_ROOT = "/import/scratch-01/jhw31/DS_10283_2791/"
    torchaudio.set_audio_backend("sox_io")
elif platform == "darwin":
    VOICEBANK_ROOT = ""
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
hparams = {'lr': 10e-6,
            'initialisation_distribution': nn.init.xavier_uniform_,
            'noise_alpha': 1,
            'speech_alpha': 1,
            'no_of_layers': 7,
            'channels': [1, 16, 32, 64, 128, 256, 256, 256],
            'lstm_layers': 2,
            'lstm_bidir': True,
            'noise_loss_type': 6,
            'speech_loss_type': 0,
            'dropout': True,
            'dropout_conv': 0.1,
            'dropout_fc': 0.2,
            'batch_size': 32,
            'optim_eps': 10e-7,
            'bound_crm_eps': 10e-7,
            'optim_weight_decay': 10e-5,
            'optim_amsgrad': True,
            'gradient_clip_val': 10.0,
            'gradient_clip_algorithm': "norm",
            'stochastic_weight_avg': True,
            'dataset_type': 28,
            'channel_attention_reduction_ratio': 16,
            'spatial_attention_kernel_size': 7}

class Config(object):
    def __init__(self):
        self.tune = False
        self.load_data_into_RAM = True
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
        self.RactivationE = nn.ReLU 
        self.RactivationD = nn.LeakyReLU
        self.CactivationE = ComplexReLU
        self.CactivationD = ComplexLReLU
        self.upsample_scale_factor = [(2,1), (2,1), (2,1), (2,1), (2,2), (2,2), (2,2)]
        self.upsampling_mode = 'nearest'
        
        self.receptive_field_freq = 291 * (self.sr / self.fft_size)
        self.receptive_field_time = 291 / (self.sr / self.hop_length)
        self.integer_win_size = int(((1000 / (self.sr / self.hop_length)) \
                                * (self.window_length / 2) / 1000) * self.sr)

        self.val_log_sample_size = 1
        self.seed = 0
config = Config()
