from dotenv import load_dotenv
load_dotenv()

import os
import ravop as R
import json
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler

from ravdl.v2 import Pytorch_Model
import torch
from transformers import GPT2Tokenizer
import random
from ravdl.v2.optimizers import AdamW

# Initialize and create graph
R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name='poem_gpt2_model', algorithm='pytorch_model', approach='distributed', gpu_required="yes")

model_op = R.model('generate_torchscript_models/pretrained_gpt2.pt')
optimizer = AdamW(lr=5e-4, eps=1e-8)
model = Pytorch_Model(model_op=model_op)
model.initialize(optimizer)

'''
The following dataset has been taken from:
https://github.com/prajwalcr/AutoCompose
'''
with open("anticipation.json", "r") as f:
    data = json.load(f)

data = [poem for poem in data if len(poem["poem"].split()) < 100]
print("Data length: ", len(data))
# data = data[:1000]
# print(len(data))

# print(data[:5])
class PoemDataset(Dataset):
    def __init__(self, poems, tokenizer, max_length=768, gpt2_type="gpt2"):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        for poem in poems:
            encodings_dict = tokenizer("<|startoftext|>"+poem["poem"]+"<|endoftext|>",
                                        truncation=True,
                                        max_length=max_length,
                                        padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

# # Loading GPT2 Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                          bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>',
                                          pad_token='<|pad|>')

# print(tokenizer.encode(“<|startoftext|> Hello World <|endoftext|>“, padding=“max_length”, max_length=10))
print(len(tokenizer))
# Finding length of maximum token in dataset
max_length = max([len(tokenizer.encode(poem["poem"])) for poem in data])
batch_size = 32
max_length = 100
dataset = PoemDataset(data, tokenizer, max_length=max_length)
# Split data into train and validation sets
train_size = int(0.9*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print("Number of samples for training = ", train_size)
print("Number of samples for validation = ", val_size)

train_dataloader = DataLoader(train_dataset,
                              sampler=RandomSampler(train_dataset),
                              batch_size=batch_size)
val_dataloader = DataLoader(val_dataset,
                            sampler=SequentialSampler(val_dataset),
                            batch_size=batch_size)


# model.resize_token_embeddings(len(tokenizer))
# Running the model on GPU

epochs = 2
warmup_steps = 1e2
sample_every = 100
print(len(train_dataloader))
print(len(train_dataset))

total_training_steps = len(train_dataloader)*epochs

for epoch_i in range(epochs):
    print(f'Beginning epoch {epoch_i+1} of {epochs}')
    proc_count = 0
    batch_count = 0
    # Labels are shifted by 1 timestep
    for step, batch in enumerate(train_dataloader):
        b_input_ids = R.t(batch[0][:,:-1].numpy())
        b_labels = R.t(batch[0][:,1:].numpy())
        b_masks = R.t(batch[1][:,:-1].numpy())
        # model.zero_grad()

        outputs = model._forward_pass(b_input_ids, padding_mask = b_masks, training=True)
                        # labels=b_labels,
                        # attention_mask=b_masks)
        loss = R.cross_entropy_loss(b_labels, outputs, ignore_index=-1, reshape_target=(-1,len(tokenizer)), reshape_label=(-1,))

        if proc_count % 1 == 0:
            model._backward_pass(loss, step=True)
            proc_count = 0
            loss.persist_op("loss_batch_{}_epoch_{}".format(batch_count, epoch_i))
            batch_count += 1
        else:
            model._backward_pass(loss, step=False)

        proc_count += 1

model.save_model(name='poem_gpt')

R.activate()
R.execute()
R.track_progress()

model = R.fetch_persisting_op('poem_gpt')
# model = torch.jit.load('persisting_ops/poem_gpt.pt')
model.eval()

@torch.no_grad()
def generate(model, idx, max_new_tokens:int, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # forward the model to get the logits for the index in the sequence
        logits = model(idx)
        # print("\nlogits: ",logits)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

bos_token=random.randint(1,30000)
num_samples = 1
x = torch.tensor([bos_token])
    
# we'll process all desired num_samples in a batch, so expand out the batch dim
x = x.expand(num_samples, -1)

# forward the model `steps` times to get samples, in a batch
y = generate(model, x, max_new_tokens=200, do_sample=True, top_k=50)

for i in range(num_samples):
    out = tokenizer.decode(y[i].cpu().squeeze(), skip_special_tokens=True)
    print('-'*80)
    print(out)

for i in range(650, 706):
    loss_op = R.fetch_persisting_op("loss_batch_{}_epoch_1".format(i))
    print("Batch ", i, " ", loss_op)