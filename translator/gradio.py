import torch
import gradio as gr
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("google/mt5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-large")
model = PeftModel.from_pretrained(model, "lora/checkpoint-55000")
model.save_pretrained("inference")
model = AutoModelForSeq2SeqLM.from_pretrained("inference", torch_dtype=torch.float16).to("cuda")    



def preprocess(input_text, prefix):
    return prefix + input_text + "<END>"
    
def postprocess(output_text):
    output_text = output_text[4:]
    output_text = output_text.split("<END>")[0]
    
    return output_text

def translate(input_text, *input_list):
    if input_list[0] == input_list[1]:
        return f"Error: Please choose just 1 direction"
    if input_list[0]:
        input_text = preprocess(input_text, "en: ")
    if input_list[1]:
        input_text = preprocess(input_text, "vi: ")
    input_ids = tokenizer(input_text, max_length=512, padding='max_length', truncation=True,  return_tensors="pt").to("cuda")
    outputs = model.generate(
        **input_ids,
        max_new_tokens=int(input_list[2]) if input_list[2] != '' else 20,
        # early_stopping=input_list[3],
        # do_sample=input_list[4],
        num_beams=input_list[3],
        # penalty_alpha=input_list[6],
        # temperature=input_list[7],
        # top_k=input_list[8],
        # top_p=input_list[9],
        use_cache=True,
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = postprocess(output_text)
    return output_text



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Group():
                    with gr.Row():
                        en2vi = gr.Checkbox(label="En -> Vi")
                        vi2en = gr.Checkbox(label="Vi -> En")
                    text_input = gr.Textbox(label="input")
                    with gr.Row():
                        clear = gr.ClearButton([text_input])               
                        trans_button = gr.Button("Translate")
            with gr.Group():
                gr.Markdown("<center>Control the length output</center>")
                with gr.Row():
                    new_tokens = gr.Textbox(label="max_new_tokens")
                    # early = gr.Checkbox(label="early_stopping")
            with gr.Group():
                gr.Markdown("<center>Control the generate strategy</center>")
                # sample = gr.Checkbox(label="do_sampling")
                num_beam = gr.Slider(label="num_beam", minimum=1, maximum=10, step=1)
                # penaty_alpha = gr.Slider(label="penaty_alpha", minimum=0, maximum=1)
            # with gr.Group():
            #     gr.Markdown("<center>Manipulation output logits</center>")
            #     with gr.Row():
            #         temp = gr.Slider(label="temperature", minimum=0, maximum=3)
            #         top_k = gr.Slider(label="top_k", minimum=0, maximum=100)
            #         top_p = gr.Slider(label="top_p", minimum=0, maximum=1)
        with gr.Column():
            output = gr.Textbox(None, label="output")
        
    input_list = [
        text_input, 
        en2vi,
        vi2en,
        new_tokens, 
        # early,
        # sample,
        num_beam,
        # penaty_alpha,
        # temp,
        # top_k,
        # top_p,
    ]

    trans_button.click(fn=translate, inputs=input_list, outputs=output)

demo.launch(share=True)