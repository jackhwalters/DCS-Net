import torch
from scipy.io.wavfile import write
from network_functions import *
from pytorch_lightning.core.lightning import LightningModule


# source https://github.com/luuuyi/CBAM.PyTorch/blob/83d3312c8c542d71dfbb60ee3a15454ba253a2b0/model/resnet_cbam.py
class RealChannelAttention(torch.nn.Module):
    def __init__(self, no_channels, reduction_ratio):
        super(RealChannelAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1, return_indices=False)
        self.fc = torch.nn.Sequential(torch.nn.Conv2d(no_channels, max(no_channels // reduction_ratio, 1), 1, bias=False),
                               torch.nn.ReLU(),
                               torch.nn.Conv2d(max(no_channels // reduction_ratio, 1), no_channels, 1, bias=False))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x) 
        avg_out_fc = self.fc(avg_out)
        max_out = self.max_pool(x)
        max_out_fc = self.fc(max_out)
        out = avg_out_fc + max_out_fc
        out = max_out_fc
        return self.sigmoid(out)

# source https://github.com/luuuyi/CBAM.PyTorch/blob/83d3312c8c542d71dfbb60ee3a15454ba253a2b0/model/resnet_cbam.py
class RealSpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size):
        super(RealSpatialAttention, self).__init__()

        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class R_NETWORK(LightningModule):
    def __init__(self, config, hparams, seed):
        super().__init__()
        pl.seed_everything(seed)
        self.config = config
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)

        self.encoder = torch.nn.ModuleList()
        self.decoder = torch.nn.ModuleList()
        self.decoder_attention = torch.nn.ModuleList()
        self.skip_attention = torch.nn.ModuleList()

        # Encoder
        self.initial_batchnorm = torch.nn.BatchNorm2d(self.hparams['channels'][0])
        for i in range(self.hparams['no_of_layers']):
            enc_block = torch.nn.Sequential(torch.nn.Conv2d(
                                                in_channels=1 if i == 0 else self.hparams['channels'][i],
                                                out_channels=self.hparams['channels'][i + 1],
                                                kernel_size=self.config.kernel_sizeE[i],
                                                stride=self.config.strideE[i],
                                                padding=self.config.paddingE[i]),
                                            torch.nn.BatchNorm2d(self.hparams['channels'][i + 1]),
                                            self.config.RactivationE())
            self.encoder.append(enc_block)

        # Latent space
        self.lstm = torch.nn.LSTM(input_size=self.hparams['channels'][5],
                        hidden_size=self.hparams['channels'][4],
                        num_layers=self.hparams['lstm_layers'],
                        bidirectional=self.hparams['lstm_bidir'],
                        batch_first=True)
        self.fc = torch.nn.Linear(self.hparams['channels'][5],
                        self.hparams['channels'][5])

        self.dropout_conv = torch.nn.Dropout(self.hparams['dropout_conv'])
        self.dropout_fc = torch.nn.Dropout(self.hparams['dropout_fc'])

        # Decoder
        for i in range(self.hparams['no_of_layers']):
            in_channels = self.hparams['channels'][self.hparams['no_of_layers'] - i]
            in_and_skip_channels = (in_channels + in_channels)
            in_channels = in_channels
            out_channels = max(self.hparams['channels'][self.hparams['no_of_layers'] - 1 - i], 1)
            if i == self.hparams['no_of_layers'] - 1:
                dec_block = torch.nn.ConvTranspose2d(
                                                in_and_skip_channels,
                                                out_channels,
                                                kernel_size=self.config.kernel_sizeD[i],
                                                stride=self.config.strideD,
                                                padding=self.config.paddingD[i])
            else:
                dec_block = torch.nn.Sequential(torch.nn.ConvTranspose2d(
                                                    in_and_skip_channels,
                                                    out_channels,
                                                    kernel_size=self.config.kernel_sizeD[i],
                                                    stride=self.config.strideD,
                                                    padding=self.config.paddingD[i]),
                                                torch.nn.BatchNorm2d(self.hparams['channels'][self.hparams['no_of_layers'] \
                                                    - 1 - i]),
                                                self.config.RactivationD())
            self.decoder.append(dec_block)

            # Skip connection attention
            skip_channel_attenion = RealChannelAttention(in_channels, self.hparams['channel_attention_reduction_ratio'])
            skip_spatial_attention = RealSpatialAttention(self.hparams['spatial_attention_kernel_size'])
            self.skip_attention.append(skip_channel_attenion)
            self.skip_attention.append(skip_spatial_attention)

            # Decoder attention
            decoder_channel_attenion = RealChannelAttention(out_channels, self.hparams['channel_attention_reduction_ratio'])
            decoder_spatial_attention = RealSpatialAttention(self.hparams['spatial_attention_kernel_size'])
            self.decoder_attention.append(decoder_channel_attenion)
            self.decoder_attention.append(decoder_spatial_attention)

        self.weights_init()

        # enable if testing CheckBatchGradient callback in Trainer
        # self.example_input_array = torch.rand(32, 256, 256, dtype=torch.cfloat)


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                self.hparams['initialisation_distribution'](m.weight)
                # print("Initialised Conv2d")
            elif isinstance(m, torch.nn.ConvTranspose2d):
                self.hparams['initialisation_distribution'](m.weight)
                # print("Initialised ConvTranspose2d")
            elif isinstance(m, torch.nn.Linear):
                self.hparams['initialisation_distribution'](m.weight)
                # print("Initialised Linear")


    def forward(self, x):
        enc_out = []

        e = self.initial_batchnorm(x.view(x.shape[0], -1, x.shape[1], x.shape[2]))
        enc_out.append(e)

        for i in range(self.hparams['no_of_layers']):
            e = self.encoder[i](enc_out[i])
            e = self.dropout_conv(e)
            enc_out.append(e)

        latent_shape = enc_out[-1].shape
        flattened = torch.flatten(e, 2, 3).permute(0, 2, 1)
        lstm_out, _ = self.lstm(flattened)
        fc_out = self.fc(lstm_out)
        fc_out = self.dropout_fc(fc_out) if self.hparams['dropout'] else fc_out
        d = fc_out.permute(0, 2, 1).reshape(latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3])

        for i in range(self.hparams['no_of_layers']):
            dec_skip_channel_attention = self.skip_attention[(i*2)](enc_out[self.hparams['no_of_layers'] - i])
            skip_ca = dec_skip_channel_attention * enc_out[self.hparams['no_of_layers'] - i]
            dec_skip_spatial_attention = self.skip_attention[(i*2)+1](skip_ca)
            skip_sa = dec_skip_spatial_attention * skip_ca
            # skip_sa = enc_out[self.hparams['no_of_layers'] - i]

            d = torch.cat((d, skip_sa), dim=1)
            self.upsample = torch.nn.Upsample(scale_factor=self.config.upsample_scale_factor[i], mode=self.config.upsampling_mode)
            d = self.upsample(d)
            d = self.decoder[i](d)
            if i != self.hparams['no_of_layers'] - 1:
                d = d * self.decoder_attention[(i*2)](d)
                d = d * self.decoder_attention[(i*2)+1](d)
            d = self.dropout_conv(d)

        net_out = torch.squeeze(d)
        net_out_bound = torch.sigmoid(net_out)
        return net_out_bound


    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(),
                    lr=self.hparams['lr'],
                    eps=self.hparams['optim_eps'],
                    weight_decay=self.hparams['optim_weight_decay'],
                    amsgrad=self.hparams['optim_amsgrad'])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=10, verbose=True)
        return {
            'optimizer': optimiser,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss' if sys.argv[1] == "dcs" or sys.argv[1] == "drs" else 'speech_loss'
        }


    def training_step(self, train_batch, batch_idx):
        if sys.argv[1] == "dcs" or sys.argv[1] == "drs":
            noise_loss, speech_loss, train_loss = train_batch_2_loss(self, train_batch, batch_idx, dtype="real")
            metrics = {'train_loss': train_loss.detach(),
                            'noise_loss': noise_loss.detach(),
                            'speech_loss': speech_loss.detach()}
        
        elif sys.argv[1] == "dc" or sys.argv[1] == "dr":
            speech_loss = train_batch_2_loss(self, train_batch, batch_idx, dtype="real")
            metrics = {'speech_loss': speech_loss.detach()}

        self.log_dict(metrics, on_epoch=True)

        if torch.any(torch.isnan(train_loss if sys.argv[1] == "dcs" or sys.argv[1] == "drs" else speech_loss)):
            print("found NaN in R train loss!")
            return None
        else: 
            return train_loss if sys.argv[1] == "dcs" or sys.argv[1] == "drs" else speech_loss


    def validation_step(self, val_batch, val_idx):
        if sys.argv[1] == "dcs" or sys.argv[1] == "drs":
            noise_loss, speech_loss, val_loss, pesq_av, stoi_av, \
                    predict_noise_audio, predict_clean_audio, \
                        noise_audio, noisy_audio, clean_audio  = \
                            val_batch_2_metric_loss(self, val_batch, val_idx, dtype="real")
            metrics = {'val_loss': val_loss.detach(),
                    'val_noise_loss': noise_loss.detach(),
                    'val_speech_loss': speech_loss.detach(),
                    'val_pesq': torch.tensor(pesq_av),
                    'val_stoi': torch.tensor(stoi_av)}
            output = {
                "clean": clean_audio.cpu().numpy(),
                "predict_clean": predict_clean_audio.cpu().numpy(),
                "noise": noise_audio.cpu().numpy(),
                "predict_noise": predict_noise_audio.cpu().numpy(),
                "noisy": noisy_audio.cpu().numpy()
            }

        elif sys.argv[1] == "dc" or sys.argv[1] == "dr":
            speech_loss, pesq_av, stoi_av, \
                predict_clean_audio, \
                    noise_audio,  noisy_audio,  clean_audio = \
                        val_batch_2_metric_loss(self, val_batch, val_idx, dtype="real")
            metrics = {'val_speech_loss': speech_loss.detach(),
                    'val_pesq': torch.tensor(pesq_av),
                    'val_stoi': torch.tensor(stoi_av)}
            output = {
                "clean": clean_audio.cpu().numpy(),
                "predict_clean": predict_clean_audio.cpu().numpy(),
                "noise": noise_audio.cpu().numpy(),
                "noisy": noisy_audio.cpu().numpy()
            }

        if torch.any(torch.isnan(val_loss if sys.argv[1] == "dcs" or sys.argv[1] == "drs" else speech_loss)):
            print("found a NaN in R val loss!")
            return None
        else:
            return output, metrics


    def validation_epoch_end(self, validation_step_outputs):
        audio = []
        metrics_list = []
        for i in range(len(validation_step_outputs)):
            audio.append(validation_step_outputs[i][0])
            metrics_list.append(validation_step_outputs[i][1])

        if sys.argv[1] == "dcs" or sys.argv[1] == "drs":
            avg_val_loss = torch.stack([x['val_loss'] for x in metrics_list]).mean()
            avg_val_noise_loss = torch.stack([x['val_noise_loss'] for x in metrics_list]).mean()
        avg_val_speech_loss = torch.stack([x['val_speech_loss'] for x in metrics_list]).mean()
        avg_val_pesq = torch.stack([x['val_pesq'] for x in metrics_list]).mean()
        avg_val_stoi = torch.stack([x['val_stoi'] for x in metrics_list]).mean()


        if sys.argv[1] == "dcs" or sys.argv[1] == "drs":
            metrics = {'val_loss': avg_val_loss, 'val_noise_loss': avg_val_noise_loss,
                    'val_speech_loss': avg_val_speech_loss, 'val_pesq': avg_val_pesq,
                    'val_stoi': avg_val_stoi, 'step': self.current_epoch}

        elif sys.argv[1] == "dc" or sys.argv[1] == "dr":
            metrics = {'val_speech_loss': avg_val_speech_loss, 'val_pesq': avg_val_pesq,
                'val_stoi': avg_val_stoi, 'step': self.current_epoch}

        epoch_end(self, audio, "val")
        self.log_dict(metrics, on_epoch=True)

        return metrics


    def test_step(self, test_batch, test_idx):
        if sys.argv[1] == "dcs" or sys.argv[1] == "drs":
            noise_loss, speech_loss, test_loss, pesq_av, stoi_av, \
                    predict_noise_audio, predict_clean_audio, \
                        noise_audio, noisy_audio, clean_audio, id, start_point = \
                            test_batch_2_metric_loss(self, test_batch, test_idx, dtype="real")
            metrics = {'test_loss': test_loss.detach(),
                    'test_noise_loss': noise_loss.detach(),
                    'test_speech_loss': speech_loss.detach(),
                    'test_pesq': torch.tensor(pesq_av),
                    'test_stoi': torch.tensor(stoi_av)}
            output = {
                "clean": clean_audio.cpu().numpy(),
                "predict_clean": predict_clean_audio.cpu().numpy(),
                "noise": noise_audio.cpu().numpy(),
                "predict_noise": predict_noise_audio.cpu().numpy(),
                "noisy": noisy_audio.cpu().numpy()
            }

        elif sys.argv[1] == "dc" or sys.argv[1] == "dr": 
            speech_loss, pesq_av, stoi_av, \
                predict_clean_audio, \
                    noise_audio,  noisy_audio,  clean_audio = \
                        test_batch_2_metric_loss(self, test_batch, test_idx, dtype="real")
            metrics = {'test_speech_loss': speech_loss.detach(),
                    'test_pesq': torch.tensor(pesq_av),
                    'test_stoi': torch.tensor(stoi_av)}
            # self.log_dict(metrics, on_epoch=True)
            output = {
                "clean": clean_audio.cpu().numpy(),
                "predict_clean": predict_clean_audio.cpu().numpy(),
                "noise": noise_audio.cpu().numpy(),
                "noisy": noisy_audio.cpu().numpy()
            }

        return output, metrics


    def test_epoch_end(self, test_step_outputs):
        audio = []
        metrics_list = []
        for i in range(len(test_step_outputs)):
            audio.append(test_step_outputs[i][0])
            metrics_list.append(test_step_outputs[i][1])

        if sys.argv[1] == "dcs" or sys.argv[1] == "drs":
            avg_test_loss = torch.stack([x['test_loss'] for x in metrics_list]).mean()
            avg_test_noise_loss = torch.stack([x['test_noise_loss'] for x in metrics_list]).mean()
        avg_test_speech_loss = torch.stack([x['test_speech_loss'] for x in metrics_list]).mean()
        avg_test_pesq = torch.stack([torch.tensor(x['test_pesq']) for x in metrics_list]).mean()
        avg_test_stoi = torch.stack([torch.tensor(x['test_stoi']) for x in metrics_list]).mean()

        if sys.argv[1] == "dcs" or sys.argv[1] == "drs":
            metrics = {'test_loss': avg_test_loss, 'test_noise_loss': avg_test_noise_loss,
                    'test_speech_loss': avg_test_speech_loss, 'test_pesq': avg_test_pesq,
                    'test_stoi': avg_test_stoi, 'step': self.current_epoch}

        elif sys.argv[1] == "dc" or sys.argv[1] == "dr":
            metrics = {'test_speech_loss': avg_test_speech_loss, 'test_pesq': avg_test_pesq,
                'test_stoi': avg_test_stoi, 'step': self.current_epoch}

        epoch_end(self, audio, "test")
        self.log_dict(metrics, on_epoch=True)

        return metrics

    
    def on_after_backward(self):
        if self.trainer.global_step % 25 == 0:
            vals = torch.tensor((), device='cuda')
            for k, v in zip(self.state_dict().keys(), self.parameters()):
                name = k
                if v.grad is None:
                    grads = 0
                else:         
                    grads = v.grad  

                grads = torch.flatten(torch.tensor(grads, device='cuda'))
                vals = torch.cat((vals, grads), 0)
            norm = torch.linalg.norm(vals)
            vals_mean = torch.mean(vals)
            self.logger.experiment.add_scalar("grad val avg", vals_mean, global_step=self.trainer.global_step)
            self.logger.experiment.add_scalar("grad norm", norm, global_step=self.trainer.global_step)
