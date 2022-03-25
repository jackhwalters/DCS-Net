import sys
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader
from torch import cuda, nn
from pytorch_lightning import Trainer, loggers as pl_loggers
from data import VoiceBankDataset, preprocess
from r_network import *
from c_network import *
from config import config, hparams
from network_functions import InputMonitor, CheckBatchGradient

partition = preprocess(config)

train_set = VoiceBankDataset(partition['train'], config, mode="train", seed=config.seed)
train_loader = DataLoader(train_set, **config.data_params)

validation_set = VoiceBankDataset(partition['val'], config, mode="val", seed=config.seed)
validation_loader = DataLoader(validation_set, **config.data_params)

if config.tune:
    def objective(trial: optuna.trial.Trial):
        hparams["lr"] = trial.suggest_float("Learning rate", 10e-6, 10e-4)
        hparams["initialisation_distribution"] = trial.suggest_categorical("Initialisation distribution",
            [nn.init.kaiming_uniform_, nn.init.xavier_uniform_])
        hparams["speech_alpha"] = trial.suggest_float("Speech alpha", 0, 1)
        hparams["lstm_layers"] = trial.suggest_int("LSTM layers", 1, 12)
        hparams["dropout_conv"] = trial.suggest_float("Convolutional dropout probability", 0.01, 0.99)
        hparams["dropout_fc"] = trial.suggest_float("Fully connected dropout probability", 0.01, 0.99)
        hparams["optim_weight_decay"] = trial.suggest_int("Optim weight decay", 10e-6, 10e-4)
        # hparams["lstm_bidir"] = trial.suggest_categorical("LSTM bidir", [True, False])
        # hparams["batch_size"] = trial.suggest_categorical("Batch size", [16, 32, 64, 128])
        if sys.argv[1] == "dcs":
            network = C_NETWORK(config, hparams, config.seed)
            if platform == "linux":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs/', name='DCS-Net-train')
            elif platform == "darwin":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DCS-Net-train')
        elif sys.argv[1] == "drs":
            network = R_NETWORK(config, hparams, config.seed)
            if platform == "linux":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs/', name='DRS-Net-train')
            elif platform == "darwin":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DRS-Net-train')
        elif sys.argv[1] == "dc":
            network = R_NETWORK(config, hparams, config.seed)
            if platform == "linux":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs/', name='DC-Net-train')
            elif platform == "darwin":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DC-Net-train')
        elif sys.argv[1] == "dr":
            network = R_NETWORK(config, hparams, config.seed)
            if platform == "linux":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs/', name='DR-Net-train')
            elif platform == "darwin":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DR-Net-train')
        else:
            print("Please pass either 'drs', 'dcs', 'dr', or 'dc' as an argument to train the desired network")

        try:
            trainer = Trainer(
                    # gpus=config.num_gpus if sys.argv[1] == "real" else 1,
                    # accelerator='ddp' if sys.argv[1] == "real" else None,
                    gpus=[int(sys.argv[2]) if platform == "linux" else 0],
                    accelerator = None,
                    max_epochs=config.max_epochs,
                    logger=tb_logger,
                    num_sanity_val_steps=config.val_log_sample_size,
                    precision=config.precision,
                    gradient_clip_val=hparams['gradient_clip_val'],
                    gradient_clip_algorithm=hparams['gradient_clip_algorithm'],
                    stochastic_weight_avg=hparams['stochastic_weight_avg'],
                    callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_pesq")])
        except IndexError:
            print("Please supply a GPU index as the 2nd command-line argument")
        
        trainer.fit(network, train_loader, validation_loader)

        return trainer.callback_metrics["val_pesq"]

    if __name__ == "__main__":
        if sys.argv[1] == "dcs":
            study_name = "dcs-net_study"
        elif sys.argv[1] == "drs":
            study_name = "drs-net_study"
        elif sys.argv[1] == "dc":
            study_name = "dc-net_study"
        elif sys.argv[1] == "dr":
            study_name = "dr-net_study"
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="maximize",
                            pruner=pruner,
                            study_name=study_name)
        study.optimize(objective,
                n_trials=100,
                timeout=None,
                show_progress_bar=True)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

elif not config.tune:
    if sys.argv[1] == "dcs":
        network = C_NETWORK(config, hparams, config.seed)
        if platform == "linux":
            tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs/', name='DCS-Net-train')
        elif platform == "darwin":
            tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DCS-Net-train')
    elif sys.argv[1] == "drs":
        network = R_NETWORK(config, hparams, config.seed)
        if platform == "linux":
            tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs/', name='DRS-Net-train')
        elif platform == "darwin":
            tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DRS-Net-train')
    elif sys.argv[1] == "dc":
        network = C_NETWORK(config, hparams, config.seed)
        if platform == "linux":
            tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs/', name='DC-Net-train')
        elif platform == "darwin":
            tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DC-Net-train')
    elif sys.argv[1] == "dr":
        network = R_NETWORK(config, hparams, config.seed)
        if platform == "linux":
            tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs/', name='DR-Net-train')
        elif platform == "darwin":
            tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DR-Net-train')
    else:
        print("Please pass either 'drs', 'dcs', 'dr', or 'dc' as an argument to train the desired network")

    # callback options: CheckBatchGradient(), InputMonitor() 
    try:
        trainer = Trainer(
                gpus=[int(sys.argv[2]) if platform == "linux" else 0],
                accelerator=None,
                max_epochs=config.max_epochs,
                logger=tb_logger,
                detect_anomaly=config.detect_anomaly,
                num_sanity_val_steps=config.val_log_sample_size,
                precision=config.precision,
                gradient_clip_val=hparams['gradient_clip_val'],
                gradient_clip_algorithm=hparams['gradient_clip_algorithm'],
                stochastic_weight_avg=hparams['stochastic_weight_avg'])
    except IndexError:
        print("Please supply a GPU index as the 2nd command-line argument")

    if __name__ == "__main__":
        trainer.fit(network, train_loader, validation_loader)
