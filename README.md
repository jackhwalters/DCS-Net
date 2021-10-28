# DCS-Net

This is the implementation of "DCS-Net: Deep Complex Subtractive Neural Network for Monaural Speech Enhancement"

## Steps to run the model

1. Edit VOICEBANK_ROOT in config.py to where your copy of the [dataset](https://datashare.ed.ac.uk/handle/10283/1942?show=full) is

2. Tune hyperparameters in config.py

3. Install the relevant modules
```
$ pip install -r requirements.txt
```

4. To run DCS-Net:
```
$ python train.py complex
```

5. To run DRS-Net:
```
$ python train.py real
```

6. To test DCS-Net:
```
$ python test.py complex
```

7. To test DRS-Net:
```
$ python test.py real
```

Example output files are available in output_files/

## Testing DC-Net or DR-Net
In order to test either DC-Net or DR-Net, switch to the speechest branch

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
