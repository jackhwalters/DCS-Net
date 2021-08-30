# DCS-Net

This is the implementation of "DCS-Net: Deep Complex Subtractive Neural Network for Monaural Speech Enhancement"

## Steps to run the model

1. Edit VOICEBANK_ROOT in config.py to where your copy of the [dataset](https://datashare.ed.ac.uk/handle/10283/1942?show=full) is

2. Tune hyperparameters in config.py

3. Install the relevant models
```
$ pip install -r requirements.txt
```

4. To run DC-Net:
```
$ python train.py complex
```

5. To run DR-Net:
```
$ python train.py real
```

6. To test DC-Net:
```
$ python test.py complex
```

7. To test DR-Net:
```
$ python test.py real
```

Example output files are available in output_files/

## Testing DCS-Net or DRS-Net
In order to test either DCS-Net or DRS-Net, switch to the master branch

## File list
data_json/ \
output_files/ \
c_network.py \
config.py \
network_functions.py \
r_network.py \
README.md \
requirements.txt \
side_tests.py \
test.py \
train.py