import json
import copy

MODEL_PATH = r'public\model\model.json'

def extract_keras_history(args):
    history = []
    if isinstance(args, list):
        for arg in args:
            if isinstance(arg, dict) and arg.get('class_name') == '__keras_tensor__':
                hist = arg.get('config', {}).get('keras_history')
                if hist:
                    history.append([hist[0], hist[1], hist[2], {}])
            elif isinstance(arg, list):
                history.extend(extract_keras_history(arg))
    return history

with open(MODEL_PATH, 'r') as f:
    m = json.load(f)

# ── Fix 1: Convert inbound_nodes from Keras 3 format to Keras 2 format ──────
cfg = m.get('modelTopology', {}).get('config', {})

def fix_layers(layers):
    for layer in layers:
        # Recurse into nested Functional models
        inner_cfg = layer.get('config', {})
        if 'layers' in inner_cfg:
            fix_layers(inner_cfg['layers'])

        nodes = layer.get('inbound_nodes', [])
        if not nodes:
            continue
        new_nodes = []
        for node in nodes:
            if isinstance(node, dict) and 'args' in node:
                history = extract_keras_history(node['args'])
                if history:
                    new_nodes.append(history)
                # else skip empty nodes (InputLayer has no inbound)
            elif isinstance(node, list):
                new_nodes.append(node)
        layer['inbound_nodes'] = new_nodes

if 'layers' in cfg:
    fix_layers(cfg['layers'])

# ── Fix 2: Add batchInputShape to InputLayer configs ─────────────────────────
def fix_input_layers(layers):
    for layer in layers:
        inner_cfg = layer.get('config', {})
        # Recurse
        if 'layers' in inner_cfg:
            fix_input_layers(inner_cfg['layers'])
        if layer.get('class_name') == 'InputLayer':
            lc = layer.get('config', {})
            if 'batch_shape' in lc and 'batchInputShape' not in lc:
                lc['batchInputShape'] = lc['batch_shape']
            if 'batch_shape' in lc and 'inputShape' not in lc:
                lc['inputShape'] = lc['batch_shape'][1:]

if 'layers' in cfg:
    fix_input_layers(cfg['layers'])

with open(MODEL_PATH, 'w') as f:
    json.dump(m, f)

print("[OK] model.json patched successfully!")
print("   - inbound_nodes converted to Keras 2 format")
print("   - InputLayer batchInputShape/inputShape added")
