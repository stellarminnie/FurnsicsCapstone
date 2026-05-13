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

with open('/home/chielsy3992/furnsics/pwa_app/public/model/model.json', 'r') as f:
    m = json.load(f)

cfg = m.get('modelTopology', {}).get('config', {})
if 'layers' in cfg:
    for layer in cfg['layers']:
        nodes = layer.get('inbound_nodes', [])
        if not nodes:
            continue
        new_nodes = []
        for node in nodes:
            if isinstance(node, dict) and 'args' in node:
                history = extract_keras_history(node['args'])
                if history:
                    new_nodes.append(history)
            elif isinstance(node, list):
                # Already in Keras 2 format
                new_nodes.append(node)
        layer['inbound_nodes'] = new_nodes

with open('/home/chielsy3992/furnsics/pwa_app/public/model/model.json', 'w') as f:
    json.dump(m, f)
print("Patched inbound_nodes in model.json")
