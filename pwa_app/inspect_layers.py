import json
with open(r'public\model\model.json', 'r', encoding='utf-8') as f:
    m = json.load(f)

top = m.get('modelTopology', {})
if 'model_config' in top:
    layers = top['model_config'].get('config', {}).get('layers', [])
else:
    layers = top.get('config', {}).get('layers', [])

for l in layers:
    if 'squeeze_excite' in l['name'] or l['class_name'] == 'GlobalAveragePooling2D':
        inbound = l.get('inbound_nodes', [])
        print(f"{l['name']} ({l['class_name']}): inbound={inbound}")
