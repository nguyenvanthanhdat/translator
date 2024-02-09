import torch
import json
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


tokenizer = AutoTokenizer.from_pretrained("google/mt5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-large", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, "lora/checkpoint-55000", torch_dtype=torch.float16).to("cuda")


def main():
    dataset = load_dataset(
        "THUDM/webglm-qa",
        split = "train",
        token = "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",
        streaming=True
    )

    dataset_head = dataset.take(10)

    num_beam = 3
    def translated(input_string):
        input_string = "en: " + input_string + "|<END>|"
        input_ids = tokenizer(input_string, max_length=512, padding='max_length', truncation=True,  return_tensors="pt").to("cuda")
        outputs = model.generate(
            **input_ids,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=num_beam,
            max_new_tokens=512,
            use_cache=True,
        )
        outputs_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return outputs_text.split("|<END>|")[0][4:]
    
    def translated_batch(input_strings):
        input_strings = ["en: " + input_string + "|<END>|" for input_string in input_strings]
        input_ids = tokenizer(input_strings, max_length=512, padding='max_length', truncation=True,  return_tensors="pt").to("cuda")
        outputs = model.generate(
            **input_ids,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=num_beam,
            max_new_tokens=512,
            use_cache=True,
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [output.split("|<END>|")[0] for output in outputs]
        outputs = [output[4:] for output in outputs]
        return outputs
    
    new_dataset = []
    for sample in tqdm(dataset_head):
        new_sample = dict()
        for i in sample:
            new_sample[i] = sample[i]
            if i != "references": 
                new_sample[f'{i}_vi'] = translated(sample[i])
            else:
                new_sample[f'{i}_vi'] = translated_batch(sample[i])
        with open(f"{num_beam}.txt", "w") as file:
            json.dump(new_dataset, file)

if __name__ == "__main__":
    main()