import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, MT5ForConditionalGeneration
from datasets import load_dataset
import wandb


# Đăng nhập vào wandb
wandb.login(key='7ac28caf9e3dc3e0685c97df182d52e13a81e311')

# Initialize WandB
wandb.init(project="msimcse-method-training", entity="hqh2042003")

# Define the embed-fusion module with a linear layer to match embedding dimensions
class EmbedFusion(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbedFusion, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, output_dim)
        self.fc_word = torch.nn.Linear(input_dim, output_dim)
        self.attention = torch.nn.MultiheadAttention(embed_dim=output_dim, num_heads=8)
        self.fc2 = torch.nn.Linear(output_dim * 2, output_dim)

    def forward(self, sentence_embedding, word_embeddings):
        sentence_embedding = self.fc1(sentence_embedding)
        word_embeddings = self.fc_word(word_embeddings)
        sentence_embedding = sentence_embedding.unsqueeze(1)
        sentence_embedding = sentence_embedding.transpose(0, 1)
        word_embeddings = word_embeddings.transpose(0, 1)
        attn_output, _ = self.attention(word_embeddings, sentence_embedding, sentence_embedding)
        added_output = attn_output + word_embeddings
        concat_output = torch.cat((added_output, sentence_embedding.expand_as(added_output)), dim=-1)
        fused_output = self.fc2(concat_output)
        return fused_output.transpose(0, 1)  # Transpose back to (batch_size, seq_length, hidden_dim)

# Load models and tokenizers
# simcse_model_name = "result_temp/xlm-roberta-large-cross_all"
simcse_model_name = "wanhin/msimcse_vi-en"
mt5_model_name = "google/mt5-base"

simcse_tokenizer = AutoTokenizer.from_pretrained(simcse_model_name)
simcse_model = AutoModel.from_pretrained(simcse_model_name)
mt5_tokenizer = AutoTokenizer.from_pretrained(mt5_model_name)
mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_model_name)

# Load dataset and select 10 samples
# dataset = load_dataset('wanhin/msimcse_512_seqlen', split='train[:10]')
dataset = load_dataset('wanhin/msimcse_512_seqlen')['train']
src_texts = dataset['sent0']
tgt_texts = dataset['sent1']

# Prepare the inputs for each sample
simcse_inputs = [simcse_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512) for text in src_texts]
mt5_inputs = [mt5_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512) for text in src_texts]
mt5_targets = [mt5_tokenizer(f"translate Vietnamese to English: {text}", return_tensors='pt', padding=True, truncation=True, max_length=512) for text in tgt_texts]

# Instantiate the embed-fusion module
input_dim = 768
output_dim = 768
embed_fusion = EmbedFusion(input_dim, output_dim)

# Define training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_fusion.to(device)
mt5_model.to(device)
simcse_model.to(device).eval()  # Set SimCSE model to evaluation mode

optimizer = torch.optim.Adam(list(embed_fusion.parameters()) + list(mt5_model.parameters()), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=mt5_tokenizer.pad_token_id)

# Log model parameters and gradients
wandb.watch(embed_fusion, log="all")
wandb.watch(mt5_model, log="all")

# Training loop
num_epochs = 50  # Set the number of epochs

for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(len(src_texts)):
        simcse_input = simcse_inputs[i]
        mt5_input = mt5_inputs[i]
        mt5_target = mt5_targets[i]

        simcse_input_ids = simcse_input['input_ids'].squeeze(1).to(device)
        mt5_input_ids = mt5_input['input_ids'].squeeze(1).to(device)
        mt5_target_ids = mt5_target['input_ids'].squeeze(1).to(device)

        # Get sentence embedding from SimCSE
        with torch.no_grad():
            simcse_outputs = simcse_model(simcse_input_ids)
            sentence_embedding = simcse_outputs.last_hidden_state.mean(dim=1)  # Average pooling

        # Get word embeddings from mT5 encoder
        encoder_outputs = mt5_model.get_encoder()(input_ids=mt5_input_ids)
        word_embeddings = encoder_outputs.last_hidden_state

        # Combine embeddings using embed-fusion module
        fused_embeddings = embed_fusion(sentence_embedding, word_embeddings)

        # Prepare attention mask
        attention_mask = mt5_input['attention_mask'].squeeze(1).to(device)

        # Forward pass through mT5 model
        outputs = mt5_model(inputs_embeds=fused_embeddings, attention_mask=attention_mask, labels=mt5_target_ids)
        loss = outputs.loss
        epoch_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log the loss and learning rate to WandB
        wandb.log({
            "loss": loss.item(),
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1,
            "step": i + 1
        })

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(src_texts)}")
    wandb.log({"epoch_loss": epoch_loss / len(src_texts), "epoch": epoch + 1})

# End WandB run
wandb.finish()