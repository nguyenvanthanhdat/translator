import os, evaluate, sys
from datasets import load_dataset

def main(lora_path):
    os.chdir(f"eval-{lora_path}")
    os.getcwd()
    bleu = evaluate.load("bleu")
    score = open("score.txt", "w")
    for file_txt in os.listdir("."):
        if file_txt.split(".")[1] == "json":
            dataset = load_dataset("json", file_txt, split='train')
        else:
            continue
        try:
            results = bleu.compute(predictions=dataset['predict'], references=dataset['label'])
        except:
            results = 0
    score.write(f"{file_txt}: bleu - {results}\n")
    score.close()
    os.chdir("..")
    os.system("zip -r result.zip eval")

if __name__ == "__main__":
    lora_path = sys.argv[1]
    main(lora_path)
