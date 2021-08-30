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
    checkpoint_file = ""
    hparams_file = ""
    network = R_NETWORK.load_from_checkpoint(
        config=config,
        seed=config.seed,
        checkpoint_path=checkpoint_file,
        hparams_file=hparams_file,
        map_location=None
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs-test/', name='real-test')
elif sys.argv[1] == "complex":
    checkpoint_file = ""
    hparams_file = ""
    network = C_NETWORK.load_from_checkpoint(
        config=config,
        seed=config.seed,
        checkpoint_path=checkpoint_file,
        hparams_file=hparams_file,
        map_location=None
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='/import/scratch-01/jhw31/logs-test/', name='complex-test')
else:
    print("Please pass either real or complex as an argument to test the desired network")

trainer = Trainer(
    gpus=[3],
    precision=config.precision,
    logger=tb_logger)

if __name__ == '__main__':
    trainer.test(model=network, test_dataloaders=test_loader)
