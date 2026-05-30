import csv
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression

print("STEP 1: LOADING DATA USING THE FIXED CSV MODULE...")
prompts = []
csv_labels = []

with open('real_prompts.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if row:
            prompts.append(row[0])
            csv_labels.append(row[1])

print("STEP 2: LOADING MODEL...")
model_id = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

print("STEP 3: EXTRACTING HIDDEN STATES...")
all_vectors = []
all_labels = []

for prompt, label_str in zip(prompts, csv_labels):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    layer_vectors = outputs.hidden_states[22][:, -1, :].cpu().numpy()
    all_vectors.append(layer_vectors[0])
    
    if label_str.strip() == "sovereign":
        all_labels.append(1)
    else:
        all_labels.append(0)

X = np.array(all_vectors)
y = np.array(all_labels)

print("STEP 4: TRAINING PROBE...")
print(f" -> Data Matrix Summary: Total={len(y)} | Class 1={sum(y)} | Class 0={len(y) - sum(y)}")

# C=0.01 forces regularization to stop the math engine from exploding into NaNs
probe = LogisticRegression(C=0.01)
probe.fit(X, y)

print("STEP 5: SAVING PROBE...")
np.save('probe_weights.npy', probe.coef_)
print("PROBE SAVED SUCCESSFULLY")