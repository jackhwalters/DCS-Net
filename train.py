import sys
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader
from torch import cuda
from pytorch_lightning import Trainer, loggers as pl_loggers
from data import VoiceBankDataset, preprocess
from r_network import *
from c_network import *
from config import Config, hparams
from network_functions import InputMonitor, CheckBatchGradient

config = Config()

partition = preprocess(config)

train_set = VoiceBankDataset(partition['train'], config, mode="train", seed=config.seed)
train_loader = DataLoader(train_set, **config.data_params)

validation_set = VoiceBankDataset(partition['val'], config, mode="val", seed=config.seed)
validation_loader = DataLoader(validation_set, **config.data_params)

if config.tune:
    def objective(trial: optuna.trial.Trial):
        # hparams["lr"] = trial.suggest_float("Learning rate", 10e-4, 10e-3)
        hparams["noise_alpha"] = trial.suggest_float("Noise alpha", 0, 1)
        hparams["speech_alpha"] = trial.suggest_float("Speech alpha", 0, 1)
        hparams["lstm_layers"] = trial.suggest_int("LSTM layers", 1, 12)
        # hparams["lstm_bidir"] = trial.suggest_categorical("LSTM bidir", [True, False])
        hparams["noise_loss_type"] = trial.suggest_int("Noise loss option", 0, 5)
        hparams["optim_weight_decay"] = trial.suggest_int("Optim weight decay", 10e-6, 10e-5)
        # hparams["speech_loss_type"] = trial.suggest_int("Speech loss option", 0, 1)
        hparams["skip_concat"] = trial.suggest_categorical("Skip concat", [True, False])
        if sys.argv[1] == "real":
            network = R_NETWORK(config, hparams, config.seed)
            if platform == "linux":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs/', name='DRS-Net-train')
            elif platform == "darwin":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DRS-Net-train')
        elif sys.argv[1] == "complex":
            network = C_NETWORK(config, hparams, config.seed)
            if platform == "linux":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs/', name='DCS-Net-train')
            elif platform == "darwin":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DCS-Net-train')
        else:
            print("Please pass either real or complex as an argument to train the desired network")

        trainer = Trainer(
                # gpus=config.num_gpus if sys.argv[1] == "real" else 1,
                # accelerator='ddp' if sys.argv[1] == "real" else None,
                gpus = [0],
                accelerator = None,
                max_epochs=config.max_epochs,
                logger=tb_logger,
                num_sanity_val_steps=config.val_log_sample_size,
                precision=config.precision,
                gradient_clip_val=hparams['gradient_clip_val'],
                gradient_clip_algorithm=hparams['gradient_clip_algorithm'],
                stochastic_weight_avg=hparams['stochastic_weight_avg'],
                callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_pesq")])
        
        trainer.fit(network, train_loader, validation_loader)

        return trainer.callback_metrics["val_pesq"]

    if __name__ == "__main__":
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="maximize",
                            pruner=pruner,
                            study_name="real_study" if sys.argv[1] == "real" else "complex_study")
        study.optimize(objective,
                n_trials=100,
                timeout=6000,
                show_progress_bar=True)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

elif not config.tune:
    if sys.argv[1] == "real":
        network = R_NETWORK(config, hparams, config.seed)
        if platform == "linux":
            tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs/', name='DRS-Net-train')
        elif platform == "darwin":
            tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DRS-Net-train')
    elif sys.argv[1] == "complex":
        network = C_NETWORK(config, hparams, config.seed)
        if platform == "linux":
            tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs/', name='DCS-Net-train')
        elif platform == "darwin":
            tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DCS-Net-train')
    else:
        print("Please pass either real or complex as an argument to train the desired network")

    # callback options: CheckBatchGradient(), InputMonitor() 
    trainer = Trainer(
            gpus=cuda.device_count(),
            accelerator = None,
            max_epochs=config.max_epochs,
            logger=tb_logger,
            num_sanity_val_steps=config.val_log_sample_size,
            precision=config.precision,
            gradient_clip_val=hparams['gradient_clip_val'],
            gradient_clip_algorithm=hparams['gradient_clip_algorithm'],
            stochastic_weight_avg=hparams['stochastic_weight_avg'])

    if __name__ == "__main__":
        trainer.fit(network, train_loader, validation_loader)
