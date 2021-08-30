import sys
from r_network import *
from c_network import *
from data import VoiceBankDataset, preprocess
from pytorch_lightning import Trainer, loggers as pl_loggers
from torch.utils.data import DataLoader
from torch import cuda
from config import Config, hparams

config = Config()

partition = preprocess(config)

test_set = VoiceBankDataset(partition['test'], config, mode="test", seed=config.seed)
test_loader = DataLoader(test_set, **config.data_params)

checkpoint_file = "lightning_logs/version_0/checkpoints/epoch=0-step=108.ckpt"
hparams_file = "lightning_logs/version_0/hparams.yaml"
if sys.argv[1] == "real":
    network = R_NETWORK(config, hparams, config.seed, checkpoint_path=checkpoint_file, hparams_file=hparams_file)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/', name='real-test')
elif sys.argv[1] == "complex":
    network = C_NETWORK(config, hparams, config.seed, checkpoint_path=checkpoint_file, hparams_file=hparams_file)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/', name='complex-test')
else:
    print("Please pass either real or complex as an argument to test the desired network")

trainer = Trainer(
        gpus=config.num_gpus if cuda.is_available() else None,
        precision=config.precision)

if __name__ == '__main__':
    trainer.test(model=network, test_dataloaders=test_loader)

