import torch
import pytorch_lightning as pl
from scipy.io.wavfile import write
from network_functions import *
from pytorch_lightning.core.lightning import LightningModule
from complexPyTorch.complexLayers import ComplexConv2d, ComplexConvTranspose2d, ComplexLinear, ComplexBatchNorm2d
from complexPyTorch.complexFunctions import complex_upsample


class C_NETWORK(LightningModule):
    def __init__(self, config, hparams, seed):
        super().__init__()
        pl.seed_everything(seed)
        self.config = config
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)

        self.encoder = torch.nn.ModuleList()
        self.decoder = torch.nn.ModuleList()

        # Encoder
        for i in range(self.hparams['no_of_layers']):
            enc_layer = ComplexConv2d(
                        in_channels=1 if i == 0 else self.hparams['channels'][i] // 2,
                        out_channels=self.hparams['channels'][i + 1] // 2,
                        kernel_size=self.config.kernel_sizeE[i],
                        stride=self.config.strideE[i],
                        padding=self.config.paddingE[i])
            self.encoder.append(enc_layer)
            enc_bn = ComplexBatchNorm2d(self.hparams['channels'][i + 1] // 2)
            self.encoder.append(enc_bn)

        # Latent space
        self.lstm = ComplexLSTM(
                        input_size=self.hparams['channels'][4],
                        hidden_size=self.hparams['channels'][4] // 2,
                        num_layers=self.hparams['lstm_layers'],
                        bidirectional=self.hparams['lstm_bidir'],
                        batch_first=True)
        self.fc = ComplexLinear(
                        self.hparams['channels'][5] // 2,
                        self.hparams['channels'][5] // 2) 

        # Decoder
        for i in range(self.hparams['no_of_layers']):
            current_layer_channels = self.hparams['channels'][self.hparams['no_of_layers'] - i]
            if i == self.hparams['no_of_layers'] - 1:
                next_layer_channels = 1
                in_channels = ((current_layer_channels + next_layer_channels) // 2) + 1
                out_channels = 1
            else:
                next_layer_channels = self.hparams['channels'][self.hparams['no_of_layers'] - 1 - i]
                in_channels = (current_layer_channels + next_layer_channels) // 2
                out_channels = next_layer_channels // 2
            dec_layer = ComplexConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=self.config.kernel_sizeD[i],
                        stride=self.config.strideD,
                        padding=self.config.paddingD[i])
            self.decoder.append(dec_layer)
            dec_bn = ComplexBatchNorm2d(self.hparams['channels'][self.hparams['no_of_layers'] - 1 - i] // 2)
            self.decoder.append(dec_bn)

        self.dropout_conv = torch.nn.Dropout(self.hparams['dropout_conv'])
        self.dropout_fc = torch.nn.Dropout(self.hparams['dropout_fc'])

        self.weights_init()

        # enable if testing CheckBatchGradient callback in Trainer
        # self.example_input_array = torch.rand(32, 256, 256, dtype=torch.cfloat)

    
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                self.config.initialisation_distribution(m.weight)
                print("Initialised Conv2d")
            elif isinstance(m, torch.nn.ConvTranspose2d):
                self.config.initialisation_distribution(m.weight)
                print("Initialised ConvTranspose2d")
            elif isinstance(m, torch.nn.Linear):
                self.config.initialisation_distribution(m.weight)
                print("Initialised Linear")


    def forward(self, x):
        enc_out = []

        enc_out.append(x.view(x.shape[0], -1, x.shape[1], x.shape[2]))
        
        for i in range(self.hparams['no_of_layers']):
            # print("enc forward i: ", i)
            e = self.encoder[i*2](enc_out[i])
            e = self.encoder[(i*2)+1](e)
            e = self.config.CactivationE(e)
            e = self.dropout_conv(torch.view_as_real(e))
            e = torch.view_as_complex(e)
            enc_out.append(e)
        
        latent_shape = enc_out[-1].shape
        flattened = torch.flatten(e, 2, 3).permute(0, 2, 1)
        lstm_out = self.lstm(flattened)
        fc_out = self.fc(lstm_out)
        fc_out = self.dropout_fc(torch.view_as_real(fc_out))
        fc_out = torch.view_as_complex(fc_out)
        d = fc_out.permute(0, 2, 1).reshape(latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3])

        for i in range(self.hparams['no_of_layers']):
            # print("dec forward i: ", i)
            d = complex_upsample(d, scale_factor=self.config.upsample_scale_factor[i],
                                        mode=self.config.upsampling_mode)
            d = torch.cat((d, enc_out[self.hparams['no_of_layers'] - 1 - i]), dim=1)
            if i == self.hparams['no_of_layers'] - 1:
                d = self.decoder[i*2](d)
            else:
                d = self.decoder[i*2](d)
                d = self.decoder[(i*2)+1](d) 
                d = self.config.CactivationD(d)
            d = self.dropout_conv(torch.view_as_real(d))
            d = torch.view_as_complex(d)

        net_out = torch.squeeze(d)
        net_out_bound = bound_cRM(net_out, self.hparams)
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
            'monitor': 'val_loss'
        }


    def training_step(self, train_batch, batch_idx):
        noise_loss, speech_loss, train_loss = train_batch_2_loss(self, train_batch, batch_idx, \
                                                                                dtype="complex")

        metrics = {'train_loss': train_loss.detach(),
                        'noise_loss': noise_loss.detach(),
                        'speech_loss': speech_loss.detach()}
        self.log_dict(metrics, on_epoch=True)

        if torch.any(torch.isnan(train_loss)):
            print("found NaN in C train loss!")
            return None
        else: 
            return train_loss


    def validation_step(self, val_batch, val_idx):
        noise_loss, speech_loss, val_loss, pesq_av, stoi_av, \
                predict_noise_audio, predict_clean_audio, \
                    noise_audio,  noisy_audio,  clean_audio = \
                        val_batch_2_metric_loss(self, val_batch, val_idx, dtype="complex")

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
        if torch.any(torch.isnan(val_loss)):
            print("found a NaN in C val loss!")
            return None
        else:
            return output, metrics


    def validation_epoch_end(self, validation_step_outputs):
        audio = []
        metrics_list = []
        for i in range(len(validation_step_outputs)):
            audio.append(validation_step_outputs[i][0])
            metrics_list.append(validation_step_outputs[i][1])

        avg_val_loss = torch.stack([x['val_loss'] for x in metrics_list]).mean()
        avg_val_noise_loss = torch.stack([x['val_noise_loss'] for x in metrics_list]).mean()
        avg_val_speech_loss = torch.stack([x['val_speech_loss'] for x in metrics_list]).mean()
        avg_val_pesq = torch.stack([x['val_pesq'] for x in metrics_list]).mean()
        avg_val_stoi = torch.stack([x['val_stoi'] for x in metrics_list]).mean()

        metrics = {'val_loss': avg_val_loss, 'val_noise_loss': avg_val_noise_loss,
                'val_speech_loss': avg_val_speech_loss, 'val_pesq': avg_val_pesq,
                'val_stoi': avg_val_stoi, 'step': self.current_epoch}

        epoch_end(self, audio, "val")
        self.log_dict(metrics, on_epoch=True)

        return metrics


    def test_step(self, test_batch, test_idx):
        noise_loss, speech_loss, test_loss, pesq_av, stoi_av, \
                predict_noise_audio, predict_clean_audio, \
                    noise_audio, noisy_audio, clean_audio, id, start_point = \
                        test_batch_2_metric_loss(self, test_batch, test_idx, dtype="complex")

        metrics = {'test_loss': test_loss,
                'test_noise_loss': noise_loss,
                'test_speech_loss': speech_loss,
                'test_pesq': pesq_av,
                'test_stoi': stoi_av}

        output = {
            "clean": clean_audio.cpu().numpy(),
            "predict_clean": predict_clean_audio.cpu().numpy(),
            "noise": noise_audio.cpu().numpy(),
            "predict_noise": predict_noise_audio.cpu().numpy(),
            "noisy": noisy_audio.cpu().numpy()
        }
        return output, metrics


    def test_epoch_end(self, test_step_outputs):
        audio = []
        metrics_list = []
        for i in range(len(test_step_outputs)):
            audio.append(test_step_outputs[i][0])
            metrics_list.append(test_step_outputs[i][1])

        avg_test_loss = torch.stack([x['test_loss'] for x in metrics_list]).mean()
        avg_test_noise_loss = torch.stack([x['test_noise_loss'] for x in metrics_list]).mean()
        avg_test_speech_loss = torch.stack([x['test_speech_loss'] for x in metrics_list]).mean()
        avg_test_pesq = torch.stack([torch.tensor(x['test_pesq']) for x in metrics_list]).float().mean()
        avg_test_stoi = torch.stack([torch.tensor(x['test_stoi']) for x in metrics_list]).mean()

        metrics = {'test_loss': avg_test_loss, 'test_noise_loss': avg_test_noise_loss,
                'test_speech_loss': avg_test_speech_loss, 'test_pesq': avg_test_pesq,
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
