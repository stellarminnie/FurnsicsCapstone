import json

path = 'public/model/model.json'
with open(path, 'r', encoding='utf-8') as f:
    m = json.load(f)

top = m['modelTopology']
if 'model_config' in top:
    layers = top['model_config']['config']['layers']
else:
    layers = top['config']['layers']

def fix_input_layers(layers):
    for l in layers:
        cfg = l.get('config', {})
        if l.get('class_name') == 'InputLayer':
            # We want ONLY batchInputShape
            if 'batch_shape' in cfg:
                if 'batchInputShape' not in cfg:
                    cfg['batchInputShape'] = cfg['batch_shape']
                del cfg['batch_shape']
            
            if 'inputShape' in cfg:
                del cfg['inputShape']
            
        if 'layers' in cfg:
            fix_input_layers(cfg['layers'])

fix_input_layers(layers)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(m, f)

print("Fixed InputLayers in model.json")
