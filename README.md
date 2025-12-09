# An-Empirical-Study-on-Deep-Learning-Based-Line-Level-Software-Defect-Prediction
## Datasets
The datasets is stored in [Dataset](line-level-defect-prediction/Dataset). They are obtained from Wattanakriengkrai et. al. The datasets contain 32 software releases across 9 software projects. The datasets that we used in our experiment can be found [here](https://github.com/awsm-research/line-level-defect-prediction.git). When reproducing the experiments for the eight models, please move or copy the data to the correct location as required by each model's structure.

## Environment Setup
Make sure R is upgraded to version 4.* or above, and CMake is upgraded to 3.15+.
### Python Environment Setup
clone the github repository by using the following command:
'''git clone https://github.com/awsm-research/DeepLineDP.git'''

use the following command to install required libraries in conda environment
'''
 conda env create -f requirements.yml
 conda activate DeepLineDP_env
'''
install PyTorch library by following the instruction from this link (the installation instruction may vary based on OS and CUDA version)

### R Environment Setup
Download the following package: tidyverse, gridExtra, ModelMetrics, caret, reshape2, pROC, effsize, ScottKnottESD

## Experiment
### Experimental Setup

We use the following hyper-parameters to train our DeepLineDP model

- `batch_size` = 32
- `num_epochs` = 10
- `embed_dim (word embedding size)` = 50
- `word_gru_hidden_dim` = 64
- `sent_gru_hidden_dim` = 64
- `word_gru_num_layers` = 1
- `sent_gru_num_layers` = 1
- `dropout` = 0.2
- `lr (learning rate)` = 0.001

### Data Preprocessing

1. run the command to prepare data for file-level model training. The output will be stored in `./datasets/preprocessed_data`
    
    ```
     python preprocess_data.py
    ```
    
2. run the command to prepare data for line-level baseline. The output will be stored in `./datasets/ErrorProne_data/` (for ErrorProne), and `./datasets/n_gram_data/` (for n-gram)
    
    ```
     python export_data_for_line_level_baseline.py
    ```
### DeepLineDP Model Training and Prediction Generation

To train DeepLineDP models, run the following command:

```
python train_model.py -dataset <DATASET_NAME>
```

The trained models will be saved in `./output/model/DeepLineDP/<DATASET_NAME>/`, and the loss will be saved in `../output/loss/DeepLineDP/<DATASET_NAME>-loss_record.csv`

To make a prediction of each software release, run the following command:

```
python generate_prediction.py -dataset <DATASET_NAME>
```

The generated output is a csv file which contains the following information:

- `project`: A software project, as specified by <DATASET_NAME>
- `train`: A software release that is used to train DeepLineDP models
- `test`: A software release that is used to make a prediction
- `filename`: A file name of source code
- `file-level-ground-truth`: A label indicating whether source code is clean or defective
- `prediction-prob`: A probability of being a defective file
- `prediction-label`: A prediction indicating whether source code is clean or defective
- `line-number`: A line number of a source code file
- `line-level-ground-truth`: A label indicating whether the line is modified
- `is-comment-line`: A flag indicating whether the line is comment
- `token`: A token in a code line
- `token-attention-score`: An attention score of a token

The generated output is stored in `./output/prediction/DeepLineDP/within-release/`

To make a prediction across software project, run the following command:

```
python generate_prediction_cross_projects.py -dataset <DATASET_NAME>
```

The generated output is a csv file which has the same information as above, and is stored in `./output/prediction/DeepLineDP/cross-project/`

python generate_prediction_cross_projects.py -dataset <DATASET_NAME>
The generated output is a csv file which has the same information as above, and is stored in ./output/prediction/DeepLineDP/cross-project/
