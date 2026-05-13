import json

path = 'public/model/model.json'
with open(path, 'r', encoding='utf-8') as f:
    m = json.load(f)

# Build a mapping of layer_name -> in_channels from weightsManifest
weight_dict = {}
for manifest in m.get('weightsManifest', []):
    for weight in manifest.get('weights', []):
        name = weight['name']
        shape = weight['shape']
        # Typically "layer_name/kernel"
        if '/kernel' in name and len(shape) == 4:
            layer_name = name.split('/')[0]
            # Conv2D kernel shape is [filter_height, filter_width, in_channels, out_channels]
            in_channels = shape[2]
            weight_dict[layer_name] = in_channels

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

# First, remove any existing "_reshape" layers we added before
inner_layers_filtered = [l for l in inner_layers if not l.get('name').endswith('_reshape')]
# Restore inbound_nodes of Conv2D layers that we previously patched
for l in inner_layers_filtered:
    if l.get('class_name') == 'Conv2D' and 'squeeze_excite_conv' in l.get('name', '') and not l.get('name').endswith('_1'):
        inbound = l.get('inbound_nodes', [])
        if inbound and len(inbound) > 0 and len(inbound[0]) > 0:
            pool_name = inbound[0][0][0]
            if pool_name.endswith('_reshape'):
                inbound[0][0][0] = pool_name.replace('_reshape', '')

new_layers = []

for l in inner_layers_filtered:
    if l.get('class_name') == 'Conv2D' and 'squeeze_excite_conv' in l.get('name', ''):
        if not l.get('name').endswith('_1'):
            layer_name = l.get('name')
            in_channels = weight_dict.get(layer_name)
            
            if not in_channels:
                print(f"Warning: Could not find in_channels for {layer_name}")
                continue
                
            inbound = l.get('inbound_nodes', [])
            if inbound and len(inbound) > 0 and len(inbound[0]) > 0:
                pool_name = inbound[0][0][0]
                reshape_name = pool_name + "_reshape"
                
                r = {
                    "class_name": "Reshape",
                    "name": reshape_name,
                    "config": {
                        "name": reshape_name,
                        "trainable": True,
                        "target_shape": [1, 1, in_channels]
                    },
                    "inbound_nodes": [[[pool_name, 0, 0, {}]]]
                }
                new_layers.append(r)
                
                inbound[0][0][0] = reshape_name

# Second, build the new layers list in topological order
final_layers = []
for l in inner_layers_filtered:
    final_layers.append(l)
    
    # If this is a layer we need to reshape, find its corresponding new reshape layer
    layer_name = l.get('name')
    for r in new_layers:
        if r['inbound_nodes'][0][0][0] == layer_name:
            final_layers.append(r)
            break

# Re-assign inner layers back to model
found_submodel = False
if 'model_config' in top:
    for l in layers:
        if l.get('name') == 'MobileNetV3Small':
            l['config']['layers'] = final_layers
            found_submodel = True
            break
else:
    for l in layers:
        if l.get('name') == 'MobileNetV3Small':
            l['config']['layers'] = final_layers
            found_submodel = True
            break

if not found_submodel:
    print("No MobileNetV3Small submodel found, treating as flattened model.")
    if 'model_config' in m['modelTopology']:
        m['modelTopology']['model_config']['config']['layers'] = final_layers
    elif 'config' in m['modelTopology']:
        m['modelTopology']['config']['layers'] = final_layers
    else:
        print("Error: Could not find layers list to write back to!")
        exit(1)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(m, f)

print(f"Successfully added {len(new_layers)} Reshape layers in topological order.")
print(f"Total layers in output: {len(final_layers)}")
