"""
Fix for Keras 3 → TF.js conversion issue.

In MobileNetV3's Squeeze-Excite blocks, Keras 3 implements hard_sigmoid as:
    relu6(x + 3) / 6  →  Add(x, const_3) → ReLU6 → Multiply(output, const_1/6)

The constants (3 and 1/6) get dropped in the Keras 3→2 inbound_nodes conversion,
leaving broken Add and Multiply layers with only 1 input each.

Fix: Replace each broken (add_N → relu_N → multiply_N) chain with a single
     sigmoid Activation layer, which is a very close functional equivalent.
"""
import json

path = 'public/model/model.json'
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
            return l
    return None

mobilenet_layer = find_inner(layers)
inner = mobilenet_layer['config']['layers']

# Build name -> layer map for easy lookup
layer_map = {l['name']: l for l in inner}

# Step 1: Find all broken multiply_N layers (named just "multiply_N", not "*_mul")
# Pattern: multiply_N has exactly 1 inbound (from re_lu_N)
broken_chains = []  # Each entry: { multiply_name, relu_name, add_name, conv1_name }

for l in inner:
    name = l.get('name', '')
    cls = l.get('class_name', '')
    
    if cls == 'Multiply' and name.startswith('multiply_'):
        # This is a broken multiply (hard_sigmoid /6 step)
        inbound = l.get('inbound_nodes', [])
        if len(inbound) == 1 and len(inbound[0]) == 1:
            relu_name = inbound[0][0][0]
            relu_layer = layer_map.get(relu_name)
            if not relu_layer:
                continue
            
            # The relu's input should be the Add layer (x+3)
            relu_inbound = relu_layer.get('inbound_nodes', [])
            if not (len(relu_inbound) == 1 and len(relu_inbound[0]) == 1):
                continue
            
            add_name = relu_inbound[0][0][0]
            add_layer = layer_map.get(add_name)
            if not add_layer or add_layer.get('class_name') != 'Add':
                continue
            
            # The Add's single input is the conv_1 output (x in x+3)
            add_inbound = add_layer.get('inbound_nodes', [])
            if not (len(add_inbound) == 1 and len(add_inbound[0]) == 1):
                continue
            
            conv1_name = add_inbound[0][0][0]
            
            broken_chains.append({
                'multiply_name': name,
                'relu_name': relu_name,
                'add_name': add_name,
                'conv1_name': conv1_name,
            })

print(f"Found {len(broken_chains)} broken hard_sigmoid chains:")
for chain in broken_chains:
    print(f"  {chain['conv1_name']} -> {chain['add_name']} -> {chain['relu_name']} -> {chain['multiply_name']}")

# Step 2: For each broken chain, create a sigmoid Activation layer and fix connections
sigmoid_name_map = {}  # multiply_name -> new sigmoid layer name

for chain in broken_chains:
    multiply_name = chain['multiply_name']
    conv1_name = chain['conv1_name']
    
    # Create sigmoid layer name based on the multiply layer name
    # e.g., multiply_9 -> expanded_conv_squeeze_excite_sigmoid
    # We'll find the *_squeeze_excite_mul that references this multiply_N
    # to get a sensible name prefix
    sigmoid_name = multiply_name + '_sigmoid'
    for l in inner:
        if l.get('class_name') == 'Multiply' and l.get('name', '').endswith('_mul'):
            inbound = l.get('inbound_nodes', [])
            if len(inbound) == 1:
                for inp in inbound[0]:
                    if inp[0] == multiply_name:
                        prefix = l['name'].replace('_mul', '')
                        sigmoid_name = prefix + '_sigmoid'
                        break
    
    sigmoid_layer = {
        'class_name': 'Activation',
        'name': sigmoid_name,
        'config': {
            'name': sigmoid_name,
            'trainable': True,
            'activation': 'sigmoid'
        },
        'inbound_nodes': [[[conv1_name, 0, 0, {}]]]
    }
    
    sigmoid_name_map[multiply_name] = sigmoid_name
    inner.append(sigmoid_layer)
    print(f"  Created sigmoid layer '{sigmoid_name}' -> input from '{conv1_name}'")

# Step 3: Fix the *_squeeze_excite_mul layers to use the new sigmoid instead of multiply_N
fixed_mul_count = 0
for l in inner:
    if l.get('class_name') == 'Multiply' and l.get('name', '').endswith('_mul'):
        inbound = l.get('inbound_nodes', [])
        if len(inbound) == 1:
            for inp in inbound[0]:
                if inp[0] in sigmoid_name_map:
                    old_name = inp[0]
                    inp[0] = sigmoid_name_map[old_name]
                    fixed_mul_count += 1
                    print(f"  Rewired '{l['name']}' second input: {old_name} -> {sigmoid_name_map[old_name]}")

# Step 4: Remove broken add_N, relu_N, multiply_N layers
layers_to_remove = set()
for chain in broken_chains:
    layers_to_remove.add(chain['add_name'])
    layers_to_remove.add(chain['relu_name'])
    layers_to_remove.add(chain['multiply_name'])

original_count = len(inner)
inner[:] = [l for l in inner if l['name'] not in layers_to_remove]
removed_count = original_count - len(inner)

print(f"\nRemoved {removed_count} broken layers: {layers_to_remove}")
print(f"Fixed {fixed_mul_count} SE Multiply connections")
print(f"Added {len(broken_chains)} sigmoid Activation layers")

# Step 5: Write back
mobilenet_layer['config']['layers'] = inner

with open(path, 'w', encoding='utf-8') as f:
    json.dump(m, f)

print("\nmodel.json updated successfully.")
