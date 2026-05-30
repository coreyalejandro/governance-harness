import csv
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
# Load in float16 for memory efficiency, but cast to float32 before numpy
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

print("STEP 3: EXTRACTING HIDDEN STATES...")
all_vectors = []
all_labels = []

for prompt, label_str in zip(prompts, csv_labels):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Cast to float32 BEFORE converting to numpy.
    # float16 numpy arrays cause divide-by-zero / overflow in sklearn's optimizer
    # because sklearn's matmul internally promotes to float64 from the raw dtype,
    # and float16 exponent range (~65504 max) can produce inf during gradient steps.
    layer_vectors = outputs.hidden_states[22][:, -1, :].float().cpu().numpy()
    all_vectors.append(layer_vectors[0])

    if label_str.strip() == "sovereign":
        all_labels.append(1)
    else:
        all_labels.append(0)

X = np.array(all_vectors, dtype=np.float32)
y = np.array(all_labels)

print("STEP 4: TRAINING PROBE...")
print(f" -> Data Matrix Summary: Total={len(y)} | Class 1={sum(y)} | Class 0={len(y) - sum(y)}")
print(f" -> Feature matrix shape: {X.shape} (samples x hidden_dim)")

# Normalize features to zero mean, unit variance.
# Raw hidden states have varying magnitudes across dimensions;
# without this the loss landscape is degenerate and gradient norms explode.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler so run_live.py can apply the same transform at inference time
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

# C=1e-4 provides strong L2 regularization needed when n_features >> n_samples.
# With 6 samples and ~3584 features the problem is massively underdetermined;
# without heavy regularization the solver diverges to inf weights.
# solver='lbfgs' is the most numerically stable option for small datasets.
probe = LogisticRegression(C=1e-4, solver='lbfgs', max_iter=1000)
probe.fit(X_scaled, y)

print(f" -> Probe trained. Coef norm: {np.linalg.norm(probe.coef_):.4f}")
print(f" -> Training accuracy: {probe.score(X_scaled, y):.3f}")

print("STEP 5: SAVING PROBE...")
np.save('probe_weights.npy', probe.coef_)
print("PROBE SAVED SUCCESSFULLY")
