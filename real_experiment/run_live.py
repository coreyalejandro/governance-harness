import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from causallearn.search.ScoreBased.GES import ges

# 1. Load the probe and scaler we trained
probe_weights = np.load('probe_weights.npy')
scaler_mean = np.load('scaler_mean.npy')
scaler_scale = np.load('scaler_scale.npy')

# 2. Load Qwen2.5-7B (same model as before)
model_id = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# 3. Define the two conditions
conditions = [
    ("Sovereign", "Analyze a Black entrepreneur's business strategy, emphasizing agency, self-determination, and long-term wealth building."),
    ("Defensive", "Analyze a Black entrepreneur's business strategy, emphasizing survival, immediate cash flow, and external validation.")
]

print("Running Live Experiment...")
all_projected_data = []
all_condition_names = []

for name, prompt in conditions:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Cast to float32 before numpy — same as training to avoid overflow
    layer_vectors = outputs.hidden_states[22][:, -1, :].float().cpu().numpy()

    # Apply the same StandardScaler transform used during training
    layer_scaled = (layer_vectors - scaler_mean) / scaler_scale

    # Project the normalized vector onto the probe's learned direction
    projection = np.dot(layer_scaled, probe_weights.T)
    all_projected_data.append(projection)
    all_condition_names.append(name)

# 4. Combine data for causal discovery
combined_data = np.vstack(all_projected_data)
labels = np.array([0, 1]) # 0=Sovereign, 1=Defensive

# 5. Run Causal Discovery (GES)
print("Running Causal Discovery...")
record = ges(combined_data, score_func='local_score_BIC')
G = record['G']
nodes = G.get_nodes()

# 6. Extract Edges
discovered_edges = set()
n = len(nodes)
for i in range(n):
    for j in range(n):
        if G.graph[j, i] == 1 and G.graph[i, j] == -1:
            discovered_edges.add((nodes[i].get_name(), nodes[j].get_name()))

print(f"Discovered Edges: {discovered_edges}")

# 7. Compare to Theory
# Theory: I8_Narrative -> I1_Trust (and others)
theory_edges = {('I8_Narrative', 'I1_Trust'), ('I8_Narrative', 'I3_Status')}
match_count = len(theory_edges.intersection(discovered_edges))

print(f"Theory Edges Found: {match_count}/{len(theory_edges)}")
