import sys
from r_network import *
from c_network import *
from data import VoiceBankDataset, preprocess
from pytorch_lightning import Trainer, loggers as pl_loggers
from torch.utils.data import DataLoader
from torch import cuda
from config import Config, hparams

config = Config()
config.data_params['batch_size'] = 1

partition = preprocess(config)

test_set = VoiceBankDataset(partition['test'], config, mode="test", seed=config.seed)
test_loader = DataLoader(test_set, **config.data_params)

if sys.argv[1] == "real":
    checkpoint_file = "/Volumes/Work/Project/Logs/logs29-8-21/real/version_0/checkpoints/epoch=131-step=38279.ckpt"
    hparams_file = "/Volumes/Work/Project/Logs/logs29-8-21/real/version_0/hparams.yaml"
    network = R_NETWORK.load_from_checkpoint(
        config=config,
        seed=config.seed,
        checkpoint_path=checkpoint_file,
        hparams_file=hparams_file,
        map_location=None
    )
    network.eval()
    if platform == "linux":
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs-test/', name='real-test')
    elif platform == "darwin":
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='real-test')

elif sys.argv[1] == "complex":
    checkpoint_file = "/Volumes/Work/Project/Logs/logs29-8-21/complex/version_0/checkpoints/epoch=168-step=49009.ckpt"
    hparams_file = "/Volumes/Work/Project/Logs/logs29-8-21/complex/version_0/hparams.yaml"
    network = C_NETWORK.load_from_checkpoint(
        config=config,
        seed=config.seed,
        checkpoint_path=checkpoint_file,
        hparams_file=hparams_file,
        map_location=None
    )
    network.eval()
    if platform == "linux":
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs-test/', name='complex-test')
    elif platform == "darwin":
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Volumes/Work/Project/Logs', name='complex-test')
else:
    print("Please pass either real or complex as an argument to test the desired network")

trainer = Trainer(
    gpus=cuda.device_count(),
    precision=config.precision,
    logger=tb_logger)

if __name__ == '__main__':
    trainer.test(model=network, test_dataloaders=test_loader)
