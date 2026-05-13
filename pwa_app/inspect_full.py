import json

path = 'public/model/model.json'
with open(path, 'r', encoding='utf-8') as f:
    m = json.load(f)

# Check top-level modelTopology structure (TF.js expects class_name at top level)
top = m.get('modelTopology', {})
print("== modelTopology top-level keys ==")
for k in top.keys():
    print(f"  {k}")

# TF.js needs class_name + config at top level, but Keras 3 uses model_config
if 'model_config' in top and 'class_name' not in top:
    print("\n[WARNING] 'class_name' missing at modelTopology level - TF.js may not be able to deserialize!")
    mc = top['model_config']
    print(f"  model_config class_name: {mc.get('class_name')}")
print()

# Collect all unique class names and activations
def walk_layers(layers, class_names, activations):
    for l in layers:
        cls = l.get('class_name', 'UNKNOWN')
        class_names.add(cls)
        cfg = l.get('config', {})
        act = cfg.get('activation')
        if act:
            if isinstance(act, str):
                activations.add(act)
            elif isinstance(act, dict):
                activations.add(str(act))
        if 'layers' in cfg:
            walk_layers(cfg['layers'], class_names, activations)

if 'model_config' in top:
    all_layers = [top['model_config']] + top['model_config'].get('config', {}).get('layers', [])
else:
    all_layers = top.get('config', {}).get('layers', [])

class_names = set()
activations = set()
walk_layers(all_layers, class_names, activations)

print("== Unique layer class_names ==")
for c in sorted(class_names):
    print(f"  {c}")

print("\n== Unique activation functions ==")
for a in sorted(activations):
    print(f"  {a}")

# Also check for any remaining non-TF.js strings
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

for keyword in ['hard_silu', 'hardSilu', 'hard_swish', 'hardSwish', 'mish', 'gelu', 'RescalingV2', 'TFSMLayer']:
    if keyword in text:
        print(f"\n[WARNING] Found unsupported keyword: {keyword}")
