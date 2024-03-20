# 1. Dataset

To down dataset test (1000 samples), please run `sh scripts/down_data.sh`

The model had trained in 2 stage:

Stage 1 Dataset with token lengths: 20 <= x <= 511 : [link](https://huggingface.co/datasets/presencesw/hash_v6)

Stage 2 Dataset phoMT: [link](https://huggingface.co/datasets/presencesw/hash_v6.5?row=1)

# 2. Fine-tune Model

Run the scripts below model

```sh
sh scripts/run.sh
```

Wandb when training 2 stage:

Stage 1: [link 1](https://wandb.ai/nguyenvanthanhdat1810/translator/runs/41bx0kze?nw=nwusernguyenvanthanhdat1810) - [link 2](https://wandb.ai/nguyenvanthanhdat1810/translator/runs/0f0edaga?nw=nwusernguyenvanthanhdat1810) (continue train)

Stage 2: [link](https://wandb.ai/nguyenvanthanhdat1810/translator/runs/2p1phaxs?nw=nwusernguyenvanthanhdat1810)

# Interface translate

To interface, first please download checkpoint lora, by run code:

```shell
sh scripts/down_lora.sh
```

Then run gradio:

```shell
python -m translator.gradio
```

# 4. Translate dataset benchmark

To Translate the dataset benchmark, such as PhoMT you can run:

```shell
CUDA_VISIBLE_DEVICES=0 python -m translator.translate --dataset_name <HF repo> --batch_size <batch_size>
```

When the dataset and model is not large and it enough to run in kaggle you can run:

```shell
MODEL_NAME_OR_PATH=<Model_name>
TOKEN_NAME=<Model_name>
HF_TOKEN=<HF_token>
DATA_TEST_PATH=<HF_repo>
SPLIT='test'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --gpu_ids 0,1 --num_processes=2 -m translator.eval \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name_or_path $TOKEN_NAME \
    --hf_key $HF_TOKEN \
    --data_test_path $DATA_TEST_PATH \
    --split $SPLIT \
    --num_proc 2 \
    --batch_size 1 \
    --max_len 512 \
    --num_beams 3,4,5 \
    --hf_key $HF_TOKEN \
    --use_lora True 
```

# 5. Use llm to evaluate the translation

## 5.1 Check the End token exist after postprocessing

The code run to "END" remain after postprocess split special token "`<EOS>`"

```shell
python -m translator.eval_translate --dataset_name <HF repo>
```

## 5.2 Evaluate the translation by llm

Use llm [(Vistral-7B)](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat) to evaluate the translation

```shell
CUDA_VISIBLE_DEVICES=0 python -m translator.eval_translate_END --dataset_name <HF repop>
```

## 6. Inference with gradio

Run the code below for fast inference:

```shell
CUDA_VISIBLE_DEVICES=0 python -m translator.gradio
```
