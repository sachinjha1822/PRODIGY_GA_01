from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load the fine-tuned GPT-2 model and tokenizer
model_name = 'gpt2'
model_path = './fine_tuned_gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_text = ""
    if request.method == 'POST':
        try:
            prompt = request.form['prompt']
            
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors='pt')
            
            if 'input_ids' in inputs and inputs['input_ids'] is not None:
                input_ids = inputs['input_ids']
                
                # Create attention_mask
                attention_mask = torch.ones_like(input_ids)

                # Generate text with adjusted parameters
                output = model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,    
                    max_length=50,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=1.0,              # Adjust temperature for diversity
                    top_p=0.95,                   # Adjust top_p for diversity
                    num_beams=5,                  # Adjust num_beams for diversity
                    early_stopping=True
                )

                # Decode the generated output into text
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            else:
                generated_text = "Error: Invalid input received."

        except Exception as e:
            generated_text = f"Error: {str(e)}"

    return render_template('index.html', generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
