# 1. Dataset

To down dataset test (1000 samples), please run `sh scripts/down_data.sh`

# 2. Fine-tune Model

Run the scripts below model

```sh
sh scripts/run.sh
```

# 3. Interface translate

To interface, first please download checkpoint lora, by run code:

```shell
sh scripts/down_lora.sh
```

Then run gradio:

```shell
python -m translator.gradio
```

# 4. Use llm to evaluate the translation

## 4.1 Check the End token exist after postprocessing

```shell
python -m translator.eval_translate --dataset_name <HF repo>
```

## 4.2 Evaluate the translation by llm

```shell
CUDA_VISIBLE_DEVICES=0 python -m translator.eval_translate_END --dataset_name <HF repop>
```
