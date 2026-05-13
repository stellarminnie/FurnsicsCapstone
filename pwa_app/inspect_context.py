import json

# Load the ORIGINAL model.json before any patches
# Since we don't have git, let's find what layers surround multiply_9
path = r'public\model\model.json'
with open(path, 'r', encoding='utf-8') as f:
    m = json.load(f)

top = m.get('modelTopology', {})
if 'model_config' in top:
    layers = top['model_config'].get('config', {}).get('layers', [])
else:
    layers = top.get('config', {}).get('layers', [])

def find_inner(layers):
    for l in layers:
        if l.get('name') == 'MobileNetV3Small':
            return l.get('config', {}).get('layers', [])
    return layers

inner = find_inner(layers)

# Find indices of the problematic multiply_N layers and show surrounding context
for i, l in enumerate(inner):
    if l.get('class_name') == 'Multiply' and l.get('name', '').startswith('multiply_'):
        # Show 3 layers before and after
        start = max(0, i-5)
        end = min(len(inner), i+3)
        print(f"\n=== Context around {l['name']} (index {i}) ===")
        for j in range(start, end):
            nl = inner[j]
            print(f"  [{j}] {nl['name']} ({nl['class_name']}): inbound={nl.get('inbound_nodes', [])}")
        break  # Just show first one for brevity
