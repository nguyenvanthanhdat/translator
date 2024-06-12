import torch
from transformers import AutoTokenizer, AutoModel, MT5ForConditionalGeneration

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


# Instantiate the embed-fusion module
input_dim = 768
output_dim = 768
embed_fusion = EmbedFusion(input_dim, output_dim)

# Define training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_fusion.to(device)
mt5_model.to(device)
simcse_model.to(device).eval()  # Set SimCSE model to evaluation mode

# Define testing function
def test_model_with_input(sentence):
    # Tokenize input sentence
    simcse_input = simcse_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    mt5_input = mt5_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

    # Get sentence embedding from SimCSE
    with torch.no_grad():
        simcse_outputs = simcse_model(simcse_input['input_ids'])
        sentence_embedding = simcse_outputs.last_hidden_state.mean(dim=1)  # Average pooling

    # Get word embeddings from mT5 encoder
    encoder_outputs = mt5_model.get_encoder()(input_ids=mt5_input['input_ids'])
    word_embeddings = encoder_outputs.last_hidden_state

    # Combine embeddings using embed-fusion module
    fused_embeddings = embed_fusion(sentence_embedding, word_embeddings)

    # Prepare attention mask
    attention_mask = mt5_input['attention_mask']

    # Test forward pass through mT5 decoder
    decoder_input_ids = mt5_tokenizer("Translate Vietnamese to English:", return_tensors='pt').input_ids.to(device)
    outputs = mt5_model.generate(
        inputs_embeds=fused_embeddings,  # Updated line
        attention_mask=attention_mask,
        max_length=512  # Set a max length for the generated output
    )

    # Decode the output tokens
    output_text = mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

# Function to test the model with input from keyboard
def interactive_test():
    while True:
        sentence = input("Enter a sentence in Vietnamese (or type 'exit' to quit): ")
        if sentence.lower() == 'exit':
            break
        output = test_model_with_input(sentence)
        print("Input:", sentence)
        print("Output:", output)
        print("\n" + "-"*50 + "\n")

# Run interactive test
interactive_test()