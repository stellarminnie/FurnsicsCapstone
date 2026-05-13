import json

path = 'public/model/model.json'
with open(path, 'r', encoding='utf-8') as f:
    m = json.load(f)

top = m.get('modelTopology', {})
if 'model_config' in top:
    layers = top['model_config'].get('config', {}).get('layers', [])
else:
    layers = top.get('config', {}).get('layers', [])

# Find MobileNetV3Small inner layers
def find_inner(layers):
    for l in layers:
        if l.get('name') == 'MobileNetV3Small':
            return l.get('config', {}).get('layers', [])
    return layers

inner = find_inner(layers)

# Show all Multiply layer inbound_nodes
for l in inner:
    if l.get('class_name') == 'Multiply':
        print(f"\nLayer: {l['name']}")
        print(f"  inbound_nodes: {json.dumps(l.get('inbound_nodes', []))}")
