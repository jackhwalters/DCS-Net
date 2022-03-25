import sys
from r_network import *
from c_network import *
from data import VoiceBankDataset, preprocess
from pytorch_lightning import Trainer, loggers as pl_loggers
from torch.utils.data import DataLoader
from torch import cuda
from config import config, hparams

config.data_params['batch_size'] = 1

partition = preprocess(config)

test_set = VoiceBankDataset(partition['test'], config, mode="test", seed=config.seed)
test_loader = DataLoader(test_set, **config.data_params)

if sys.argv[1] == "dcs":
    checkpoint_file = "/import/scratch-01/jhw31/logs/DCS-Net-train/version_0/checkpoints/epoch=0-step=289.ckpt"
    hparams_file = "/import/scratch-01/jhw31/logs/DCS-Net-train/version_0/hparams.yaml"
    network = C_NETWORK.load_from_checkpoint(
        config=config,
        seed=config.seed,
        checkpoint_path=checkpoint_file,
        hparams_file=hparams_file,
        map_location=None
    )
    network.eval()
    if platform == "linux":
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs-test/', name='DCS-Net-test')
    elif platform == "darwin":
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DCS-Net-test')

elif sys.argv[1] == "drs":
    checkpoint_file = "/import/scratch-01/jhw31/logs/DRS-Net-train/version_0/checkpoints/epoch=1-step=579.ckpt"
    hparams_file = "/import/scratch-01/jhw31/logs/DRS-Net-train/version_0/hparams.yaml"
    network = R_NETWORK.load_from_checkpoint(
        config=config,
        seed=config.seed,
        checkpoint_path=checkpoint_file,
        hparams_file=hparams_file,
        map_location=None
    )
    network.eval()
    if platform == "linux":
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs-test/', name='DRS-Net-test')
    elif platform == "darwin":
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DRS-Net-test')

elif sys.argv[1] == "dc":
    checkpoint_file = "/import/scratch-01/jhw31/logs/DC-Net-train/version_0/checkpoints/epoch=3-step=1159.ckpt"
    hparams_file = "/import/scratch-01/jhw31/logs/DC-Net-train/version_0/hparams.yaml"
    network = C_NETWORK.load_from_checkpoint(
        config=config,
        seed=config.seed,
        checkpoint_path=checkpoint_file,
        hparams_file=hparams_file,
        map_location=None
    )
    network.eval()
    if platform == "linux":
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs-test/', name='DC-Net-test')
    elif platform == "darwin":
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DC-Net-test')

elif sys.argv[1] == "dr":
    checkpoint_file = "/import/scratch-01/jhw31/logs/DR-Net-train/version_0/checkpoints/epoch=15-step=4639.ckpt"
    hparams_file = "/import/scratch-01/jhw31/logs/DR-Net-train/version_0/hparams.yaml"
    network = R_NETWORK.load_from_checkpoint(
        config=config,
        seed=config.seed,
        checkpoint_path=checkpoint_file,
        hparams_file=hparams_file,
        map_location=None
    )
    network.eval()
    if platform == "linux":
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs-test/', name='DR-Net-test')
    elif platform == "darwin":
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='DR-Net-test')

else:
    print("Please pass either real or complex as an argument to test the desired network")

trainer = Trainer(
    gpus=[int(sys.argv[2])],
    accelerator=None,
    precision=config.precision,
    logger=tb_logger)

if __name__ == '__main__':
    trainer.test(model=network, test_dataloaders=test_loader)
