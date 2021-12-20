import torch
import csv
import pytorch_lightning as pl
from sys import platform
if platform == "linux":
    from pypesq import pesq
elif platform == "darwin":
    from pesq import pesq
    from pesq.cypesq import NoUtterancesError
from pystoi import stoi
from math import isnan, pi
from numpy import random

def check_inf_neginf_nan(tensor, error_msg):
    assert not torch.any(torch.isinf(tensor)), error_msg
    if tensor.dtype == torch.complex32 or tensor.dtype == torch.complex64 or tensor.dtype == torch.complex128:
        assert not torch.any(torch.isneginf(tensor.real)), error_msg
        assert not torch.any(torch.isneginf(tensor.imag)), error_msg
    else:
        assert not torch.any(torch.isneginf(tensor)), error_msg
    assert not torch.any(torch.isnan(tensor)), error_msg

def l2_norm(s1, s2):
    norm = torch.sum(s1*s2, -1, keepdim=True)

    return norm

# source https://arxiv.org/pdf/2008.00264.pdf
class SiSNR(object):
    def __call__(self, clean, estimate, eps=1e-8):
        dot = l2_norm(estimate, clean)
        norm = l2_norm(clean, clean)

        s_target =  (dot * clean)/(norm+eps)
        e_nosie = estimate - s_target

        target_norm = l2_norm(s_target, s_target)
        noise_norm = l2_norm(e_nosie, e_nosie)
        snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)

        return torch.mean(snr)

# source https://github.com/chanil1218/DCUnet.pytorch/blob/2dcdd30804be47a866fde6435cbb7e2f81585213/train.py
class wSDR(object):
    def __call__(self, mixed, clean, clean_est, eps=2e-8):
        bsum = lambda x: torch.sum(x, dim=1)
        def mSDRLoss(orig, est):
            correlation = bsum(orig * est)
            energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
            return -(correlation / (energies + eps))

        noise = mixed - clean
        noise_est = mixed - clean_est

        a = bsum(clean**2) / (bsum(clean**2) + bsum(noise**2) + eps)
        target_wSDR = a * mSDRLoss(clean, clean_est)
        noise_wSDR = (1 - a) * mSDRLoss(noise, noise_est)
        wSDR = target_wSDR + noise_wSDR 
        return torch.mean(wSDR)

def cRM(S, Y, eps=1e-8):
    M_r_numer = (Y.real * S.real) + (Y.imag * S.imag)
    M_r_denom = torch.square(Y.real) + torch.square(Y.imag)
    M_r = M_r_numer / (M_r_denom + eps)

    M_i_numer = (Y.real * S.imag) - (Y.imag * S.real)
    M_i_denom = torch.square(Y.real) + torch.square(Y.imag)
    M_i = M_i_numer / (M_i_denom + eps)

    M = torch.complex(M_r, M_i)

    return M

def bound_cRM(cRM, hparams):
    target_noise_mask_mag = torch.abs(cRM)
    target_noise_mask_mag_tanh = torch.tanh(target_noise_mask_mag)
    target_noise_mag_tanh_real = target_noise_mask_mag_tanh * torch.cos(torch.angle(cRM))
    target_noise_mag_tanh_imag = target_noise_mask_mag_tanh * torch.sin(torch.angle(cRM))
    target_noise_mask_phase = torch.atan2(target_noise_mag_tanh_imag + hparams['bound_crm_eps'], \
        target_noise_mag_tanh_real + hparams['bound_crm_eps'])

    target_noise_mask_real = target_noise_mask_mag_tanh * torch.cos(target_noise_mask_phase)
    target_noise_mask_imag = target_noise_mask_mag_tanh * torch.sin(target_noise_mask_phase)

    return torch.complex(target_noise_mask_real, target_noise_mask_imag)

def complex_mat_mult(A, B):
    outp_real = (A.real * B.real) - (A.imag * B.imag)
    outp_imag = (A.real * B.imag) + (A.imag * B.real)

    Y = torch.complex(outp_real, outp_imag)

    return Y

def complex_lrelu(input):
    # return torch.nn.functional.leaky_relu(input.real) + 1j*torch.nn.functional.leaky_relu(input.imag)
    return torch.complex(torch.nn.functional.leaky_relu(input.real), torch.nn.functional.leaky_relu(input.imag))

def apply_complex(fr, fi, input):
    # return (fr(input.real)[0]-fi(input.imag)[0]) + 1j*(fr(input.imag)[0]+fi(input.real)[0])
    return torch.complex(fr(input.real)-fi(input.imag), (fr(input.imag)+fi(input.real)))

# source https://github.com/huyanxin/DeepComplexCRN/blob/bc6fd38b0af9e8feb716c81ff8fbacd7f71ad82f/complexnn.py
class ComplexLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, batch_first, projection_dim=None):
        super(ComplexLSTM, self).__init__()

        self.input_dim = input_size
        self.rnn_units = hidden_size
        self.real_lstm = torch.nn.LSTM(input_size=self.input_dim, hidden_size=self.rnn_units, num_layers=num_layers,
                        bidirectional=bidirectional, batch_first=batch_first)
        self.imag_lstm = torch.nn.LSTM(input_size=self.input_dim, hidden_size=self.rnn_units, num_layers=num_layers,
                        bidirectional=bidirectional, batch_first=batch_first)
        if bidirectional:
            bidirectional=2
        else:
            bidirectional=1
        if projection_dim is not None:
            self.projection_dim = projection_dim 
            self.r_trans = torch.nn.Linear(self.rnn_units*bidirectional, self.projection_dim)
            self.i_trans = torch.nn.Linear(self.rnn_units*bidirectional, self.projection_dim)
        else:
            self.projection_dim = None

    def forward(self, inputs):
        if isinstance(inputs,list):
            real, imag = inputs.real, inputs.imag 
        elif isinstance(inputs, torch.Tensor):
            real, imag = inputs.real, inputs.imag 
        r2r_out = self.real_lstm(real)[0]
        r2i_out = self.imag_lstm(real)[0]
        i2r_out = self.real_lstm(imag)[0]
        i2i_out = self.imag_lstm(imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out 
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)
        return torch.complex(real_out, imag_out)
    
    def flatten_parameters(self):
        self.imag_lstm.flatten_parameters()
        self.real_lstm.flatten_parameters()

def mag_phase_2_wave(mag, phase, config):
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    comp = torch.complex(real, imag)
    comp = torch.nn.functional.pad(comp, (0,0,0,1))
    
    audio = torch.istft(comp, n_fft=config.fft_size, hop_length=config.hop_length, \
            win_length=config.window_length, normalized=config.normalise_stft)

    return audio

def calc_metric(clean_audio, predict_audio, config, metric):
    metric_arr = []
    for i in range(predict_audio.shape[0]):
        try:
            if platform == "darwin" and metric.__name__ == "pesq":
                metric_i = metric(config.sr, clean_audio[i,:].cpu().numpy(), predict_audio[i,:].cpu().numpy(), 'wb')
            else:
                metric_i = metric(clean_audio[i,:].cpu().numpy(), predict_audio[i,:].cpu().numpy(), config.sr)
            if not isnan(metric_i):
                metric_arr.append(metric_i)
        except NoUtterancesError:
            print("Got a NoUtterancesError")
    metric_av = float(sum(metric_arr)) / max(len(metric_arr), 1)

    return metric_av

def calc_loss(self, target_noise_mask, predict_noise_mask, \
                predict_noise_audio, predict_clean_audio,
                noise_audio, noisy_audio, clean_audio):

    if self.hparams['noise_loss_type'] == 0:
        noise_loss_orig = self.config.L1(target_noise_mask, predict_noise_mask)
    elif self.hparams['noise_loss_type'] == 1:
        noise_loss_orig = self.config.wSDR(noisy_audio, noise_audio, predict_noise_audio)
    elif self.hparams['noise_loss_type'] == 2:
        noise_loss_orig = self.config.L1(target_noise_mask, predict_noise_mask) + \
            self.config.L1(noise_audio, predict_noise_audio)
    elif self.hparams['noise_loss_type'] == 3:
        noise_loss_orig = self.config.wSDR(noisy_audio, noise_audio, predict_noise_audio) + \
            self.config.L1(noise_audio, predict_noise_audio)
    elif self.hparams['noise_loss_type'] == 4:
        noise_loss_orig = self.config.wSDR(noisy_audio, noise_audio, predict_noise_audio) + \
            self.config.L1(target_noise_mask, predict_noise_mask)
    elif self.hparams['noise_loss_type'] == 5:
        if target_noise_mask.dtype == torch.complex32 or target_noise_mask.dtype == torch.complex64 or target_noise_mask.dtype == torch.complex128:
            noise_loss_orig = self.config.wSDR(noisy_audio, noise_audio, predict_noise_audio) + \
            self.config.mse(target_noise_mask.real, predict_noise_mask.real) + \
            self.config.mse(target_noise_mask.imag, predict_noise_mask.imag)
        else:
            noise_loss_orig = self.config.wSDR(noisy_audio, noise_audio, predict_noise_audio) + \
                self.config.mse(target_noise_mask, predict_noise_mask)
    noise_loss = (self.hparams['noise_alpha'] * noise_loss_orig)

    if self.hparams['speech_loss_type'] == 0:
        speech_loss_orig = -self.config.SiSNR(clean_audio, predict_clean_audio)
    speech_loss = (self.hparams['speech_alpha'] * speech_loss_orig)
 
    total_loss = noise_loss + speech_loss

    return noise_loss, speech_loss, total_loss

def train_batch_2_loss(self, train_batch, batch_idx, dtype):
    noise_data, noisy_data, clean_data, id = train_batch

    noise_mag = torch.abs(noise_data)
    noise_phase = torch.angle(noise_data)
    noisy_mag = torch.abs(noisy_data)
    noisy_phase = torch.angle(noisy_data)
    clean_mag = torch.abs(clean_data)
    clean_phase = torch.angle(clean_data)
    noise_audio = mag_phase_2_wave(noise_mag, noise_phase, self.config)
    noisy_audio = mag_phase_2_wave(noisy_mag, noisy_phase, self.config)
    clean_audio = mag_phase_2_wave(clean_mag, clean_phase, self.config)
    
    if dtype == "real":
        target_noise_mask = torch.sigmoid(noise_mag / noisy_mag)

        noisy_mag_scaled = (noisy_mag - self.config.data_minR) / (self.config.data_maxR - self.config.data_minR)
        predict_noise_mask = self(noisy_mag_scaled)

        predict_noise_mag = noisy_mag * predict_noise_mask
        predict_clean_mag = noisy_mag - predict_noise_mag
        predict_noise_audio = mag_phase_2_wave(predict_noise_mag, noisy_phase, self.config)
        predict_clean_audio = mag_phase_2_wave(predict_clean_mag, noisy_phase, self.config)
    
    elif dtype == "complex":
        target_noise_mask_out = cRM(noise_data, noisy_data)
        target_noise_mask = bound_cRM(target_noise_mask_out, self.hparams)

        # noisy_data_standardised = (noisy_data - torch.mean(noisy_data)) / torch.std(noisy_data)
        noisy_data_scaled = torch.view_as_complex((2 * ((torch.view_as_real(noisy_data) - self.config.data_minC) /
                    (self.config.data_maxC - self.config.data_minC))) - 1)
        predict_noise_mask_out = self(noisy_data_scaled)
        predict_noise_mask = bound_cRM(predict_noise_mask_out, self.hparams)

        predict_noise_data = complex_mat_mult(noisy_data, predict_noise_mask)
        predict_clean_data = noisy_data - predict_noise_data
        predict_noise_audio = mag_phase_2_wave(torch.abs(predict_noise_data), \
                                torch.angle(predict_noise_data), self.config)
        predict_clean_audio = mag_phase_2_wave(torch.abs(predict_clean_data), \
                                torch.angle(predict_clean_data), self.config)

    noise_loss, speech_loss, train_loss = calc_loss(self,
                                                    target_noise_mask=target_noise_mask,
                                                    predict_noise_mask=predict_noise_mask,
                                                    predict_noise_audio=predict_noise_audio,
                                                    predict_clean_audio=predict_clean_audio,
                                                    noise_audio=noise_audio,
                                                    noisy_audio=noisy_audio,
                                                    clean_audio=clean_audio)

    return noise_loss, speech_loss, train_loss

def val_batch_2_metric_loss(self, val_batch, val_idx, dtype):
    noise_data, noisy_data, clean_data, id = val_batch

    noise_mag = torch.abs(noise_data)
    noise_phase = torch.angle(noise_data)
    noisy_mag = torch.abs(noisy_data)
    noisy_phase = torch.angle(noisy_data)
    clean_mag = torch.abs(clean_data)
    clean_phase = torch.angle(clean_data)
    noise_audio = mag_phase_2_wave(noise_mag, noise_phase, self.config)
    noisy_audio = mag_phase_2_wave(noisy_mag, noisy_phase, self.config)
    clean_audio = mag_phase_2_wave(clean_mag, clean_phase, self.config)

    if dtype == "real":
        target_noise_mask = torch.sigmoid(noise_mag / noisy_mag)

        noisy_mag_scaled = (noisy_mag - self.config.data_minR) / (self.config.data_maxR - self.config.data_minR)
        predict_noise_mask = self(noisy_mag_scaled)
        
        predict_noise_mag = noisy_mag * predict_noise_mask
        predict_clean_mag = noisy_mag - predict_noise_mag
        
        predict_clean_audio = mag_phase_2_wave(predict_clean_mag, noisy_phase, self.config)
        predict_noise_audio = mag_phase_2_wave(predict_noise_mag, noisy_phase, self.config)

    elif dtype == "complex":
        target_noise_mask_out = cRM(noise_data, noisy_data)
        target_noise_mask = bound_cRM(target_noise_mask_out, self.hparams)

        # noisy_data_standardised = (noisy_data - torch.mean(noisy_data)) / torch.std(noisy_data)
        noisy_data_scaled = torch.view_as_complex((2 * ((torch.view_as_real(noisy_data) - self.config.data_minC) /
                    (self.config.data_maxC - self.config.data_minC))) - 1)
        predict_noise_mask_out = self(noisy_data_scaled)
        predict_noise_mask = bound_cRM(predict_noise_mask_out, self.hparams)

        predict_noise_data = complex_mat_mult(noisy_data, predict_noise_mask)
        predict_clean_data = noisy_data - predict_noise_data
        predict_clean_audio = mag_phase_2_wave(torch.abs(predict_clean_data), \
                                torch.angle(predict_clean_data), self.config)
        predict_noise_audio = mag_phase_2_wave(torch.abs(predict_noise_data), \
                                torch.angle(predict_noise_data), self.config)

    pesq_av = calc_metric(clean_audio, predict_clean_audio, self.config, pesq)
    stoi_av = calc_metric(clean_audio, predict_clean_audio, self.config, stoi)

    noise_loss, speech_loss, val_loss = calc_loss(self,
                                                    target_noise_mask=target_noise_mask,
                                                    predict_noise_mask=predict_noise_mask,
                                                    predict_noise_audio=predict_noise_audio,
                                                    predict_clean_audio=predict_clean_audio,
                                                    noise_audio=noise_audio,
                                                    noisy_audio=noisy_audio,
                                                    clean_audio=clean_audio)

    return noise_loss, speech_loss, val_loss, pesq_av, stoi_av, \
                    predict_noise_audio, predict_clean_audio, \
                    noise_audio, noisy_audio, clean_audio   

def test_batch_2_metric_loss(self, test_batch, test_idx, dtype):
    noise_data, noisy_data, clean_data, id, start_point = test_batch

    noise_mag = torch.abs(noise_data)
    noise_phase = torch.angle(noise_data)
    noisy_mag = torch.abs(noisy_data)
    noisy_phase = torch.angle(noisy_data)
    clean_mag = torch.abs(clean_data)
    clean_phase = torch.angle(clean_data)
    noise_audio = mag_phase_2_wave(noise_mag, noise_phase, self.config)
    noisy_audio = mag_phase_2_wave(noisy_mag, noisy_phase, self.config)
    clean_audio = mag_phase_2_wave(clean_mag, clean_phase, self.config)

    if dtype == "real":
        target_noise_mask = torch.sigmoid(noise_mag / noisy_mag)

        noisy_mag_scaled = (noisy_mag - self.config.data_minR) / (self.config.data_maxR - self.config.data_minR)
        predict_noise_mask = self(noisy_mag_scaled)

        predict_noise_mag = noisy_mag * predict_noise_mask
        predict_clean_mag = noisy_mag - predict_noise_mag
        
        predict_clean_audio = mag_phase_2_wave(predict_clean_mag, noisy_phase, self.config)
        predict_noise_audio = mag_phase_2_wave(predict_noise_mag, noisy_phase, self.config)

    elif dtype == "complex":
        target_noise_mask_out = cRM(noise_data, noisy_data)
        target_noise_mask = bound_cRM(target_noise_mask_out, self.hparams)

        # noisy_data_standardised = (noisy_data - torch.mean(noisy_data)) / torch.std(noisy_data)
        noisy_data_scaled = torch.view_as_complex((2 * ((torch.view_as_real(noisy_data) - self.config.data_minC) /
                    (self.config.data_maxC - self.config.data_minC))) - 1) 
        predict_noise_mask_out = self(noisy_data_scaled)
        predict_noise_mask = bound_cRM(predict_noise_mask_out, self.hparams)

        predict_noise_data = complex_mat_mult(noisy_data, predict_noise_mask)
        predict_clean_data = noisy_data - predict_noise_data
        predict_clean_audio = mag_phase_2_wave(torch.abs(predict_clean_data), \
                                torch.angle(predict_clean_data), self.config)
        predict_noise_audio = mag_phase_2_wave(torch.abs(predict_noise_data), \
                                torch.angle(predict_noise_data), self.config)

    noise_audio = mag_phase_2_wave(noise_mag, noise_phase, self.config)
    noisy_audio = mag_phase_2_wave(noisy_mag, noisy_phase, self.config)

    pesq_av = calc_metric(clean_audio, predict_clean_audio, self.config, pesq)
    stoi_av = calc_metric(clean_audio, predict_clean_audio, self.config, stoi)

    noise_loss, speech_loss, test_loss = calc_loss(self,
                                                    target_noise_mask=target_noise_mask,
                                                    predict_noise_mask=predict_noise_mask,
                                                    predict_noise_audio=predict_noise_audio,
                                                    predict_clean_audio=predict_clean_audio,
                                                    noise_audio=noise_audio,
                                                    noisy_audio=noisy_audio,
                                                    clean_audio=clean_audio)

    return noise_loss, speech_loss, test_loss, pesq_av, stoi_av, \
                    predict_noise_audio, predict_clean_audio, \
                    noise_audio, noisy_audio, clean_audio, id, start_point

def epoch_end(self, outputs, type):

    no_of_batches = len(outputs)
    random_batches = random.choice(no_of_batches, size=min(self.config.val_log_sample_size, no_of_batches), replace=False)

    no_of_samples = min(self.config.data_params['batch_size'],
                    outputs[-1]['clean'].shape[0],
                    outputs[-1]['predict_clean'].shape[0],
                    outputs[-1]['noise'].shape[0],
                    outputs[-1]['predict_noise'].shape[0],
                    outputs[-1]['noisy'].shape[0])
    random_samples = random.choice(no_of_samples, size=min(self.config.val_log_sample_size, no_of_samples), replace=False)

    for i, ridx in enumerate(range(min(self.config.val_log_sample_size, no_of_samples))):
        clean_sample = outputs[random_batches[ridx]]['clean'][random_samples[ridx],:]
        predict_clean_sample = outputs[random_batches[ridx]]['predict_clean'][random_samples[ridx],:]
        noise_sample = outputs[random_batches[ridx]]['noise'][random_samples[ridx],:]
        predict_noise_sample = outputs[random_batches[ridx]]['predict_noise'][random_samples[ridx],:]
        noisy_sample = outputs[random_batches[ridx]]['noisy'][random_samples[ridx],:]

        self.logger.experiment.add_audio("clean({})/{}".format(type, i),
                                        clean_sample,
                                        self.global_step,
                                        sample_rate=self.config.sr)
        self.logger.experiment.add_audio("predict_clean({})/{}".format(type, i),
                                        predict_clean_sample,
                                        self.global_step,
                                        sample_rate=self.config.sr)
        self.logger.experiment.add_audio("noise({})/{}".format(type, i),
                                        noise_sample,
                                        self.global_step,
                                        sample_rate=self.config.sr)
        self.logger.experiment.add_audio("predict_noise({})/{}".format(type, i),
                                        predict_noise_sample,
                                        self.global_step,
                                        sample_rate=self.config.sr)
        self.logger.experiment.add_audio("noisy({})/{}".format(type, i),
                                        noisy_sample,
                                        self.global_step,
                                        sample_rate=self.config.sr)

class InputMonitor(pl.Callback):

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            noise_real = batch[0].real
            noise_imag = batch[0].imag
            noisy_real = batch[1].real
            noisy_imag = batch[1].imag
            clean_real = batch[2].real
            clean_imag = batch[2].imag
            logger = trainer.logger
            logger.experiment.add_histogram("noise data real", noise_real, global_step=trainer.global_step)
            logger.experiment.add_histogram("noise data imag", noise_imag, global_step=trainer.global_step)
            logger.experiment.add_histogram("noisy data real", noisy_real, global_step=trainer.global_step)
            logger.experiment.add_histogram("noisy data imag", noisy_imag, global_step=trainer.global_step)
            logger.experiment.add_histogram("clean data real", clean_real, global_step=trainer.global_step)
            logger.experiment.add_histogram("clean data imag", clean_imag, global_step=trainer.global_step)

class CheckBatchGradient(pl.Callback):
    def on_train_start(self, trainer, model):
        n = 0

        example_input = model.example_input_array.to(model.device)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)
        output[n].abs().sum().backward()
        
        zero_grad_inds = list(range(example_input.size(0)))
        zero_grad_inds.pop(n)
        
        if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")