# zalo_ai_math

This repo is used for Track [Elementary Maths Solving](https://challenge.zalo.ai/portal/elementary-maths-solving)

Our baseline uses [mT0](https://huggingface.co/bigscience/mt0-large) as the backbone for pre-finetune and finetune stage training.
![zaloai](https://github.com/tien-ngnvan/zalo_ai_math/assets/98959709/e5a1c14c-0b86-4f62-9697-700a83a94f01)

## Install package

```
conda create -n zalo_env python==3.8 -y
conda activate zalo_env
pip install -r requirements.txt
```

## Dataset

- Firstly, we collect datasets from [FlanT5-v2](https://github.com/google-research/FLAN/tree/main/flan/v2/cot_data) and [CoT-collection](https://huggingface.co/datasets/kaist-ai/CoT-Collection). For the Vietnamese CoT dataset, we use [vinai translation](https://github.com/VinAIResearch/VinAI_Translate) and follow their hyperparameters generation to translate English data files. All of the raw files used in two stages are put in [`raw data`](https://drive.google.com/drive/folders/1Qj2rqcUtPOSIGi4r4zPp4HOOXHEhTRyn?usp=sharing) folder (processed with clean text, structure and translated).
- Secondly, for all tuning strategies in the pre-finetune and finetune stages we `only use raw dataset file this folder`. The script data builds our template for both training found at [data transform](#data-transform) (these scripts will update the latest strategies during the tuning model (hyperparameters and prompting) following the limit timeline of the Zalo AI Challenge.

Download and unzip `raw data` as our structure follows:

```
data
|__ raw
|    |__ pre-finetune
|    |__ finetune
|__ interim
|    |__ pre-finetune
|    |__ finetune
|__ .gitignore
scripts
requirements.txt
...
```

### Data transform

#### Pre-finetune stage

At this stage, we train for multilingual, so we need to create and combine the vi and en datasets for this stage.

Run `bash scripts/create_data_stage_1.sh` to create the pre-finetune dataset.

Or, if you want a specific language dataset, run `python zalo_ai_math\features\stage_1\create_{language}.py` with language is the dataset language you want in the root folder.

## Pre-finetune stage

For pre-finetune, the main ideas follow the technique of [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/pdf/2301.13688.pdf) and [Distilling Chain-of-Thought Reasoning from code-davinci-002 to FlanT5](https://arxiv.org/pdf/2301.12726.pdf). Training model please use

```bash
bash scripts/finetune_stage_1
```

## Finetune stage

Our implementation follows [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301) and [Cross-lingual Prompting: Improving Zero-shot Chain-of-Thought Reasoning across Languages](https://arxiv.org/pdf/2310.14799.pdf) which use two losses control learning steps. Run the training script to training model

```bash
bash scripts/finetune_stage_2
```

## Evaluate

Update soon

## Inference

Update soon

## Contact

If you have any questions/suggestions feel free to open an issue or send general ideas through email.

- Contact person:

  Tien Nguyen tien.ngnvan@gmail.com

  Dat Nguyen 20520436@gm.uit.edu.vn

> This repository contains experimental research and developments purpose of giving additional background details on Track Elementary Maths Solving [Zalo AI Challenge 2023](https://challenge.zalo.ai/)
