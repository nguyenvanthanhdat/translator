# Dataset

To down dataset test (1000 samples), please run `sh scripts/down_data.sh`

# Prepocess Data

The step to change dataset from 2 columns ['en', 'vi'] to ['inputs', 'targets'] 2 direction vi -> en and en -> vi

```python
python -m translator.features.finetune.create_sts
```

# Fine-tune Model

Run the scripts below model

```sh
sh scripts/train_translate.sh
```

# Interface translate

To interface, first please download checkpoint lora, by run code:

```shell
sh scripts/down_lora.sh
```

Then run gradio:

```python
python -m translator.gradio
```
