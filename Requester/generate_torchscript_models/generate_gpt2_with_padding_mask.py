import torch
from transformers import GPT2Tokenizer
from mingpt.model_with_padding_mask import GPT
from mingpt.utils import set_seed
import time
set_seed(64)

model_type = 'gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(model_type,
                                          bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>',
                                          pad_token='<|pad|>')

model = GPT.from_pretrained(model_type, tokenizer_length=len(tokenizer))

# print(len(tokenizer))

model_script = torch.jit.script(model)

model_script.save('pretrained_gpt2.pt')