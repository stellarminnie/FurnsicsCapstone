import json

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

def fix_layers(layers):
    for layer in layers:
        cfg = layer.get('config', {})
        
        # 1. Fix InputLayer shapes
        if layer.get('class_name') == 'InputLayer':
            if 'batch_shape' in cfg:
                cfg['batchInputShape'] = cfg['batch_shape']
                del cfg['batch_shape']
            if 'inputShape' in cfg:
                del cfg['inputShape'] # TF.js only wants batchInputShape or inputShape, not both

        # 2. Fix inbound_nodes
        nodes = layer.get('inbound_nodes', [])
        if nodes:
            new_nodes = []
            for node in nodes:
                if isinstance(node, dict) and 'args' in node:
                    history = extract_keras_history(node['args'])
                    if history:
                        new_nodes.append(history)
                elif isinstance(node, list):
                    new_nodes.append(node)
            layer['inbound_nodes'] = new_nodes
            
        # Recurse into nested models
        if 'layers' in cfg:
            fix_layers(cfg['layers'])

with open('public/model/model.json', 'r', encoding='utf-8') as f:
    m = json.load(f)

# Find root layers array
top = m.get('modelTopology', {})
if 'model_config' in top:
    layers = top['model_config'].get('config', {}).get('layers', [])
else:
    layers = top.get('config', {}).get('layers', [])

fix_layers(layers)

with open('public/model/model.json', 'w', encoding='utf-8') as f:
    json.dump(m, f)

print("model.json successfully patched for TF.js compatibility.")
