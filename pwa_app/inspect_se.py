import json
with open(r'public\model\model.json', 'r', encoding='utf-8') as f: m = json.load(f)
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

inner_layers = find_inner(layers)

for l in inner_layers:
    if 'squeeze_excite_pool' in l['name'] or 'squeeze_excite_conv' in l['name']:
        print(f"{l['name']} ({l['class_name']}): inbound={l.get('inbound_nodes')}")
