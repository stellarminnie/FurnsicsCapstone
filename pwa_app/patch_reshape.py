import json

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

inner_layers = find_inner(layers)
new_layers = []

for l in inner_layers:
    if l.get('class_name') == 'Conv2D' and 'squeeze_excite_conv' in l.get('name', ''):
        # We only want the first conv in the SE block, not the second one (e.g. not _conv_1)
        if not l.get('name').endswith('_1'):
            # This is the first Conv2D. Its input is the GlobalAveragePooling2D.
            inbound = l.get('inbound_nodes', [])
            if inbound and len(inbound) > 0 and len(inbound[0]) > 0:
                pool_name = inbound[0][0][0]
                reshape_name = pool_name + "_reshape"
                
                # Create the Reshape layer
                r = {
                    "class_name": "Reshape",
                    "name": reshape_name,
                    "config": {
                        "name": reshape_name,
                        "trainable": True,
                        "target_shape": [1, 1, -1]
                    },
                    "inbound_nodes": [[[pool_name, 0, 0, {}]]]
                }
                new_layers.append(r)
                
                # Point the Conv2D to the Reshape layer
                inbound[0][0][0] = reshape_name

# Add all the new Reshape layers to the inner_layers array
inner_layers.extend(new_layers)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(m, f)

print(f"Added {len(new_layers)} Reshape layers for Squeeze-Excite blocks.")
