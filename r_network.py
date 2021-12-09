import torch
from network_functions import *
from random import randint
from pytorch_lightning.core.lightning import LightningModule


class R_NETWORK(LightningModule):
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
            enc_layer = torch.nn.Conv2d(
                        1 if i == 0 else self.hparams['channels'][i],
                        self.hparams['channels'][i + 1],
                        kernel_size=self.config.kernel_sizeE[i],
                        stride=self.config.strideE[i],
                        padding=self.config.paddingE[i])
            self.encoder.append(enc_layer)
            enc_bn = torch.nn.BatchNorm2d(self.hparams['channels'][i + 1])
            self.encoder.append(enc_bn)

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
            current_layer_channels = self.hparams['channels'][self.hparams['no_of_layers'] - i] 
            if i == self.hparams['no_of_layers'] - 1:
                next_layer_channels = 1
                in_channels = ((current_layer_channels + next_layer_channels))
                out_channels = 1
            else:
                next_layer_channels = self.hparams['channels'][self.hparams['no_of_layers'] - 1 - i]
                in_channels = (current_layer_channels + next_layer_channels)
                out_channels = next_layer_channels
            dec_layer = torch.nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=self.config.kernel_sizeD[i],
                        stride=self.config.strideD,
                        padding=self.config.paddingD[i])
            self.decoder.append(dec_layer)
            dec_bn = torch.nn.BatchNorm2d(self.hparams['channels'][self.hparams['no_of_layers'] - 1 - i])
            self.decoder.append(dec_bn)

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
            e = self.encoder[i*2](enc_out[i])
            e = self.encoder[(i*2)+1](e)
            e = self.config.RactivationE(e)
            e = self.dropout_conv(e)
            enc_out.append(e)

        latent_shape = enc_out[-1].shape
        flattened = torch.flatten(e, 2, 3).permute(0, 2, 1)
        lstm_out, _ = self.lstm(flattened)
        fc_out = self.fc(lstm_out)
        fc_out = self.dropout_fc(fc_out) if self.hparams['dropout'] else fc_out
        d = fc_out.permute(0, 2, 1).reshape(latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3])

        for i in range(self.hparams['no_of_layers']):
            self.upsample = torch.nn.Upsample(scale_factor=self.config.upsample_scale_factor[i], mode=self.config.upsampling_mode)
            d = self.upsample(d)
            d = torch.cat((d, enc_out[self.hparams['no_of_layers'] - 1 - i]), dim=1)
            if i == self.hparams['no_of_layers'] - 1:
                d = self.decoder[i*2](d)
            else:
                d = self.decoder[i*2](d)
                d = self.decoder[(i*2)+1](d) 
                d = self.config.RactivationD(d)
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
            'monitor': 'speech_loss'
        }


    def training_step(self, train_batch, batch_idx):
        speech_loss = train_batch_2_loss(self, train_batch, batch_idx, dtype="real")

        metrics = {'speech_loss': speech_loss.detach()}
        self.log_dict(metrics, on_epoch=True)
        if torch.any(torch.isnan(speech_loss)):
            print("found NaN in R train loss!")
            return None
        else: 
            return speech_loss


    def validation_step(self, val_batch, val_idx):
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
        if torch.any(torch.isnan(speech_loss)):
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

        avg_val_speech_loss = torch.stack([x['val_speech_loss'] for x in metrics_list]).mean()
        avg_val_pesq = torch.stack([x['val_pesq'] for x in metrics_list]).mean()
        avg_val_stoi = torch.stack([x['val_stoi'] for x in metrics_list]).mean()

        metrics = {'val_speech_loss': avg_val_speech_loss, 'val_pesq': avg_val_pesq,
                'val_stoi': avg_val_stoi, 'step': self.current_epoch}

        epoch_end(self, audio, "val")
        self.log_dict(metrics, on_epoch=True)

        return metrics


    def test_step(self, test_batch, test_idx):
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

        avg_test_speech_loss = torch.stack([x['test_speech_loss'] for x in metrics_list]).mean()
        avg_test_pesq = torch.stack([x['test_pesq'] for x in metrics_list]).mean()
        avg_test_stoi = torch.stack([x['test_stoi'] for x in metrics_list]).mean()

        metrics = {'test_speech_loss': avg_test_speech_loss, 'test_pesq': avg_test_pesq,
                'test_stoi': avg_test_stoi, 'step': self.current_epoch}

        epoch_end(self, audio, "test")
        self.log_dict(metrics, on_epoch=True)

        return metrics

    
    def on_after_backward(self):
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
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
