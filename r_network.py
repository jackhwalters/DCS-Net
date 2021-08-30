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

        self.conv_encode1 = torch.nn.Conv2d(1, self.hparams['channels'][0], 
                            kernel_size=self.config.kernel_sizeE[0],
                            stride=self.config.strideE1,
                            padding=self.config.paddingE[0])
        self.bne1 = torch.nn.BatchNorm2d(self.hparams['channels'][0])
        self.conv_encode2 = torch.nn.Conv2d(self.hparams['channels'][0], self.hparams['channels'][1],
                            kernel_size=self.config.kernel_sizeE[1],
                            stride=self.config.strideE1,
                            padding=self.config.paddingE[1])
        self.bne2 = torch.nn.BatchNorm2d(self.hparams['channels'][1])
        self.conv_encode3 = torch.nn.Conv2d(self.hparams['channels'][1], self.hparams['channels'][2],
                            kernel_size=self.config.kernel_sizeE[2],
                            stride=self.config.strideE1,
                            padding=self.config.paddingE[2])
        self.bne3 = torch.nn.BatchNorm2d(self.hparams['channels'][2])
        self.conv_encode4 = torch.nn.Conv2d(self.hparams['channels'][2], self.hparams['channels'][3],
                            kernel_size=self.config.kernel_sizeE[3],
                            stride=self.config.strideE2,
                            padding=self.config.paddingE[3])
        self.bne4 = torch.nn.BatchNorm2d(self.hparams['channels'][3])
        self.conv_encode5 = torch.nn.Conv2d(self.hparams['channels'][3], self.hparams['channels'][4],
                            kernel_size=self.config.kernel_sizeE[4],
                            stride=self.config.strideE2,
                            padding=self.config.paddingE[4])
        self.bne5 = torch.nn.BatchNorm2d(self.hparams['channels'][4])
        self.conv_encode6 = torch.nn.Conv2d(self.hparams['channels'][4], self.hparams['channels'][4],
                            kernel_size=self.config.kernel_sizeE[5],
                            stride=self.config.strideE2,
                            padding=self.config.paddingE[5])
        self.bne6 = torch.nn.BatchNorm2d(self.hparams['channels'][4])
        self.conv_encode7 = torch.nn.Conv2d(self.hparams['channels'][4], self.hparams['channels'][4],
                            kernel_size=self.config.kernel_sizeE[6],
                            stride=self.config.strideE2,
                            padding=self.config.paddingE[6])
        self.bne7 = torch.nn.BatchNorm2d(self.hparams['channels'][4])

        self.lstm = torch.nn.LSTM(input_size=self.hparams['channels'][4],
                            hidden_size=self.hparams['channels'][3],
                            num_layers=self.hparams['lstm_layers'],
                            bidirectional=self.hparams['lstm_bidir'],
                            batch_first=True)
        self.fc = torch.nn.Linear(self.hparams['channels'][4],
                            self.hparams['channels'][4])

        self.upsample1 = torch.nn.Upsample(scale_factor=self.config.scale_factor1, mode=self.config.upsampling_mode)
        self.conv_decode1 = torch.nn.ConvTranspose2d((self.hparams['channels'][4] + self.hparams['channels'][4])
                            if self.hparams['skip_concat'] else self.hparams['channels'][4],
                            self.hparams['channels'][4],
                            kernel_size=self.config.kernel_sizeD[0],
                            stride=self.config.strideD,
                            padding=self.config.paddingD[0])
        self.bnd1 = torch.nn.BatchNorm2d(self.hparams['channels'][4])
        self.upsample2 = torch.nn.Upsample(scale_factor=self.config.scale_factor1, mode=self.config.upsampling_mode)
        self.conv_decode2 = torch.nn.ConvTranspose2d((self.hparams['channels'][4] + self.hparams['channels'][4])
                            if self.hparams['skip_concat'] else self.hparams['channels'][4],
                            self.hparams['channels'][4],
                            kernel_size=self.config.kernel_sizeD[1],
                            stride=self.config.strideD,
                            padding=self.config.paddingD[1])
        self.bnd2 = torch.nn.BatchNorm2d(self.hparams['channels'][4])
        self.upsample3 = torch.nn.Upsample(scale_factor=self.config.scale_factor1, mode=self.config.upsampling_mode)
        self.conv_decode3 = torch.nn.ConvTranspose2d((self.hparams['channels'][4] + self.hparams['channels'][3])
                            if self.hparams['skip_concat'] else self.hparams['channels'][4],
                            self.hparams['channels'][3],
                            kernel_size=self.config.kernel_sizeD[2],
                            stride=self.config.strideD,
                            padding=self.config.paddingD[2])
        self.bnd3 = torch.nn.BatchNorm2d(self.hparams['channels'][3])
        self.upsample4 = torch.nn.Upsample(scale_factor=self.config.scale_factor1, mode=self.config.upsampling_mode)
        self.conv_decode4 = torch.nn.ConvTranspose2d((self.hparams['channels'][3] + self.hparams['channels'][2])
                            if self.hparams['skip_concat'] else self.hparams['channels'][3],
                            self.hparams['channels'][2],
                            kernel_size=self.config.kernel_sizeD[3],
                            stride=self.config.strideD,
                            padding=self.config.paddingD[3])
        self.bnd4 = torch.nn.BatchNorm2d(self.hparams['channels'][2])
        self.upsample5 = torch.nn.Upsample(scale_factor=self.config.scale_factor2, mode=self.config.upsampling_mode)
        self.conv_decode5 = torch.nn.ConvTranspose2d((self.hparams['channels'][2] + self.hparams['channels'][1])
                            if self.hparams['skip_concat'] else self.hparams['channels'][2],
                            self.hparams['channels'][1],
                            kernel_size=self.config.kernel_sizeD[4],
                            stride=self.config.strideD,
                            padding=self.config.paddingD[4])
        self.bnd5 = torch.nn.BatchNorm2d(self.hparams['channels'][1])
        self.upsample6 = torch.nn.Upsample(scale_factor=self.config.scale_factor2, mode=self.config.upsampling_mode)
        self.conv_decode6 = torch.nn.ConvTranspose2d((self.hparams['channels'][1] + self.hparams['channels'][0])
                            if self.hparams['skip_concat'] else self.hparams['channels'][1],
                            self.hparams['channels'][0],
                            kernel_size=self.config.kernel_sizeD[5],
                            stride=self.config.strideD,
                            padding=self.config.paddingD[5])
        self.bnd6 = torch.nn.BatchNorm2d(self.hparams['channels'][0])
        self.upsample7 = torch.nn.Upsample(scale_factor=self.config.scale_factor2, mode=self.config.upsampling_mode)
        self.conv_decode7 = torch.nn.ConvTranspose2d((self.hparams['channels'][0] + 1)
                            if self.hparams['skip_concat'] else self.hparams['channels'][0],
                            1,
                            kernel_size=self.config.kernel_sizeD[6],
                            stride=self.config.strideD,
                            padding=self.config.paddingD[6])
        self.bnd7 = torch.nn.BatchNorm2d(1)

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
        net_in = x.view(x.shape[0], -1, x.shape[1], x.shape[2])
        e1 = self.bne1(self.config.RactivationE(self.conv_encode1(net_in)))
        e1 = self.dropout_conv(e1) if self.hparams['dropout'] else e1
        e2 = self.bne2(self.config.RactivationE(self.conv_encode2(e1)))
        e2 = self.dropout_conv(e2) if self.hparams['dropout'] else e2
        e3 = self.bne3(self.config.RactivationE(self.conv_encode3(e2)))
        e3 = self.dropout_conv(e3) if self.hparams['dropout'] else e3
        e4 = self.bne4(self.config.RactivationE(self.conv_encode4(e3)))
        e4 = self.dropout_conv(e4) if self.hparams['dropout'] else e4
        e5 = self.bne5(self.config.RactivationE(self.conv_encode5(e4)))
        e5 = self.dropout_conv(e5) if self.hparams['dropout'] else e5
        e6 = self.bne6(self.config.RactivationE(self.conv_encode6(e5)))
        e6 = self.dropout_conv(e6) if self.hparams['dropout'] else e6
        e7 = self.bne7(self.config.RactivationE(self.conv_encode7(e6)))
        e7 = self.dropout_conv(e7) if self.hparams['dropout'] else e7

        latent_shape = e7.shape
        flattened = torch.flatten(e7, 2, 3).permute(0, 2, 1)
        lstm_out, _ = self.lstm(flattened)
        fc_out = self.fc(lstm_out)
        fc_out = self.dropout_fc(fc_out) if self.hparams['dropout'] else fc_out
        unflattened = fc_out.permute(0, 2, 1).reshape(latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3])

        d1_skip = torch.cat((self.upsample1(unflattened), e6), dim=1) \
                if self.hparams['skip_concat'] else self.upsample(unflattened + e7)
        d1 = self.config.RactivationD(self.bnd1(self.conv_decode1(d1_skip)))
        d1 = self.dropout_conv(d1) if self.hparams['dropout'] else d1
        d2_skip = torch.cat((self.upsample2(d1), e5), dim=1) \
                if self.hparams['skip_concat'] else self.upsample(d1 + e6)
        d2 = self.config.RactivationD(self.bnd2(self.conv_decode2(d2_skip)))
        d2 = self.dropout_conv(d2) if self.hparams['dropout'] else d2
        d3_skip = torch.cat((self.upsample3(d2), e4), dim=1) \
                if self.hparams['skip_concat'] else self.upsample(d2 + e5)
        d3 = self.config.RactivationD(self.bnd3(self.conv_decode3(d3_skip)))
        d3 = self.dropout_conv(d3) if self.hparams['dropout'] else d3
        d4_skip = torch.cat((self.upsample4(d3), e3), dim=1) \
                if self.hparams['skip_concat'] else self.upsample(d3 + e4)
        d4 = self.config.RactivationD(self.bnd4(self.conv_decode4(d4_skip)))
        d4 = self.dropout_conv(d4) if self.hparams['dropout'] else d4
        d5_skip = torch.cat((self.upsample5(d4), e2), dim=1) \
                if self.hparams['skip_concat'] else self.upsample(d4 + e3)
        d5 = self.config.RactivationD(self.bnd5(self.conv_decode5(d5_skip)))
        d5 = self.dropout_conv(d5) if self.hparams['dropout'] else d5
        d6_skip = torch.cat((self.upsample6(d5), e1), dim=1) \
                if self.hparams['skip_concat'] else self.upsample(d5 + e2)
        d6 = self.config.RactivationD(self.bnd6(self.conv_decode6(d6_skip)))
        d6 = self.dropout_conv(d6) if self.hparams['dropout'] else d6
        d7_skip = torch.cat((self.upsample7(d6), net_in), dim=1) \
                if self.hparams['skip_concat'] else self.upsample(d6 + e1)
        d7 = self.conv_decode7(d7_skip)

        net_out = torch.squeeze(d7)

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
