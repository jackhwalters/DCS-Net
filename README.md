# DCS-Net

This is the implementation of "DCS-Net: Deep Complex Subtractive Neural Network for Monaural Speech Enhancement"

### Setup

1. Edit VOICEBANK_ROOT in config.py to be the root of the [dataset](https://datashare.ed.ac.uk/handle/10283/2791)

2. Install the dependencies
```
$ pip install -r requirements.txt
```

3. Adjust hyperparameters in config.py


### Training
{DCS-Net: dcs, DRS-Net: drs, DC-Net: dc, DR-Net: dr}
```
$ python train.py [NETWORK]
```

### Testing
{DCS-Net: dcs, DRS-Net: drs, DC-Net: dc, DR-Net: dr}
```
$ python test.py [NETWORK]
```
