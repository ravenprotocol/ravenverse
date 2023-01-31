from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import ravop as R
from transformer import GPT
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from ravdl.v2.optimizers import Adam


R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name='transformer', algorithm='transformer', approach='distributed')


optimizer = Adam()

hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
hf_model.config.pad_token_id = hf_model.config.eos_token_id # suppress a warning

sd_hf = hf_model.state_dict()

model = GPT(vocab_size=50257, embed_dim=768, 
            block_size=1024, n_heads=12, 
            n_layer=12, seq_length=21, 
            optimizer=optimizer)

keys = [k for k in sd_hf if not k.endswith(('attn.masked_bias', '.attn.bias'))]

transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

for k in keys:
    key_parts = k.split('.')
    if key_parts[0] == 'transformer':
        var_name = key_parts[1]
        if var_name == 'wte' or var_name == 'wpe':
            model.__dict__[var_name].initial_W = sd_hf[k].numpy().tolist()
        
        elif var_name == 'h':
            block_num = int(key_parts[2])
            block = model.__dict__[var_name][block_num]
            
            if key_parts[3] == 'attn':
                if key_parts[4] == 'c_attn':
                    if key_parts[5] == 'weight':
                        combined_weight = sd_hf[k].numpy()
                        block.__dict__['attn'].__dict__['wq'].initial_W = combined_weight[:,:768].T.tolist()
                        block.__dict__['attn'].__dict__['wk'].initial_W = combined_weight[:,768:2*768].T.tolist()
                        block.__dict__['attn'].__dict__['wv'].initial_W = combined_weight[:,2*768:].T.tolist()
                    elif key_parts[5] == 'bias':
                        combined_bias = sd_hf[k].numpy()
                        block.__dict__['attn'].__dict__['wq'].initial_w0 = combined_bias[:768].tolist()
                        block.__dict__['attn'].__dict__['wk'].initial_w0 = combined_bias[768:2*768].tolist()
                        block.__dict__['attn'].__dict__['wv'].initial_w0 = combined_bias[2*768:].tolist()
                
                elif key_parts[4] == 'c_proj':
                    if key_parts[5] == 'weight':
                        block.__dict__['attn'].__dict__['c_proj'].initial_W = sd_hf[k].numpy().T.tolist()
                    elif key_parts[5] == 'bias':
                        block.__dict__['attn'].__dict__['c_proj'].initial_w0 = sd_hf[k].numpy().tolist()

            
            elif key_parts[3] == 'ln_1' or key_parts[3] == 'ln_2':
                if key_parts[4] == 'weight':
                    block.__dict__[key_parts[3]].initial_W = sd_hf[k].numpy().tolist()
                elif key_parts[4] == 'bias':
                    block.__dict__[key_parts[3]].initial_w0 = sd_hf[k].numpy().tolist()

            elif key_parts[3] == 'mlp':
                if key_parts[5] == 'weight':
                    block.__dict__[key_parts[4]].initial_W = sd_hf[k].numpy().T.tolist()
                elif key_parts[5] == 'bias':
                    block.__dict__[key_parts[4]].initial_w0 = sd_hf[k].numpy().tolist()

        elif var_name == 'ln_f':
            if key_parts[2] == 'weight':
                model.__dict__[var_name].initial_W = sd_hf[k].numpy().tolist()
            elif key_parts[2] == 'bias':
                model.__dict__[var_name].initial_w0 = sd_hf[k].numpy().tolist()
    elif key_parts[0] == 'lm_head':
        var_name = key_parts[0]
        model.__dict__[var_name].initial_W = sd_hf[k].numpy().tolist()
        

def generate(prompt='', num_samples=10, steps=20, do_sample=True):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if prompt == '': 
        prompt = '<|endoftext|>'
    encoded_input = tokenizer(prompt, return_tensors='pt').to('cpu')
    x = encoded_input['input_ids']
    sequence_length = x.shape[1]
    # model.seq_length = sequence_length
    x = x.expand(num_samples, -1)
    x = x.numpy()
    x = R.t(x)
    
    causal_mask = np.triu(np.ones((sequence_length,sequence_length)),k=1)

    for i in range(steps):
        outputs = model._forward_pass(x, training=False, batch_size=1, mask=causal_mask)
        output_tensor = R.index(outputs, indices=str({"indices":"[:,-1,:]"}))
        output_token = R.argmax(output_tensor, axis=-1, keepdims="True")
        x = R.concat(x,output_token,axis=-1)
        model.seq_length = model.seq_length + 1
        causal_mask = np.triu(np.ones((model.seq_length,model.seq_length)),k=1)

    outputs.persist_op('outputs')


prompt = 'Lionel Messi was recently crowned the World Champion. His rival Cristiano Ronaldo was furious and tried to'
steps = 5
generate(prompt=prompt, num_samples=1, steps=steps)


R.activate()
R.execute()
R.track_progress()



outputs = R.fetch_persisting_op('outputs')['result']

outputs = outputs.detach().numpy()
outputs = np.argmax(outputs, axis=-1)

import torch
outputs = torch.tensor(outputs)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

final_output = prompt


for i in range(1):
    outputs = outputs[i].cpu().squeeze()[-steps:]
    out = tokenizer.decode(outputs)
    print('\n\n')
    print('-'*80)
    out = out.split(' ')
    final_output += ' '.join(out)

print('\nPrompt: ', prompt)
print('\nGPT: ', final_output, '\n')

