# gdown 1Z7g-pkRAk2Rk5a9AhxyUWHuywQaMwLyD
# gdown 1Zdk_eyB1LlOFAlFWwr2PWbQz8G2B2Ojd
gdown 1prTTLZX6AbGP2n3Epm0y4ujDBx7UNj13

mkdir data
# python -m zipfile -e PhoMT.zip data/craw
# python -m zipfile -e VietAI.zip data/craw/VietAI
python -m  zipfile -e dataset_1000.zip data/craw/finetune

# rm PhoMT.zip
# rm VietAI.zip
rm dataset_1000.zip