# Panoptic One-Click Segmentation

This is code for our paper:

```
Panoptic One-Click Segmentation: Applied to Agricultural Data
```

Please note: All steps in here need to be run from the main directory or adapted accordingly as desired.


## Downloading the data

### Datasets
The 10% data subsets of SB20 and CN20 used in the paper are included in this repository.

### Models
The original models trained and described in the paper can be downloaded here:
'''
https://uni-bonn.sciebo.de/s/d0hBDlJwIq8e3fk
'''
They need to be put in ./results/paper_models/ in order to run evaluation using the provided eval script (see below).


## Environment set-up

### Create, activate, and install packages in your the virtual environment
Create your virtual environment:
```
python3 -m venv venv
. ./venv/bin/activate
pip install --upgrade pip
pip install numpy==1.19.5
pip install -r ./requirements.txt
```


## Running the code

### Evaluation
These commands run evaluation for:
/*:
  - Basic Panoptic One-Click Segmentation model for SB20
  - Basic Panoptic One-Click Segmentation model for CN20
  - Model from Ablation Study on missing clicks (0% missing clicks)
  - Model from Ablation Study on missing clicks (25% missing clicks)
  - Model from Ablation Study on missing clicks (50% missing clicks)
  - Model from Ablation Study on missing clicks (75% missing clicks)
  - Model from Ablation Study on missing clicks (100% missing clicks)
 */

'''
./run.sh EVAL BASIC SB20
./run.sh EVAL BASIC SB20
./run.sh EVAL MISSING_CLICKS SB20 000
./run.sh EVAL MISSING_CLICKS SB20 025
./run.sh EVAL MISSING_CLICKS SB20 050
./run.sh EVAL MISSING_CLICKS SB20 075
./run.sh EVAL MISSING_CLICKS SB20 100
'''

### Training
These commands run training for:
/*:
  - Basic Panoptic One-Click Segmentation on SB20
  - Basic Panoptic One-Click Segmentation on CN20
  - Ablation study on missing clicks (0% missing clicks)
  - Ablation study on missing clicks (25% missing clicks)
  - Ablation study on missing clicks (50% missing clicks)
  - Ablation study on missing clicks (75% missing clicks)
  - Ablation study on missing clicks (100% missing clicks)
 */

'''
./run.sh TRAIN BASIC SB20
./run.sh TRAIN BASIC SB20
./run.sh TRAIN MISSING_CLICKS SB20 000
./run.sh TRAIN MISSING_CLICKS SB20 025
./run.sh TRAIN MISSING_CLICKS SB20 050
./run.sh TRAIN MISSING_CLICKS SB20 075
./run.sh TRAIN MISSING_CLICKS SB20 100
'''
