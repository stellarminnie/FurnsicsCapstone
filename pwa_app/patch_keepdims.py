import json

path = 'public/model/model.json'
with open(path, 'r', encoding='utf-8') as f:
    m = json.load(f)

top = m.get('modelTopology', {})
if 'model_config' in top:
    layers = top['model_config'].get('config', {}).get('layers', [])
else:
    layers = top.get('config', {}).get('layers', [])

def fix_keepdims(layers):
    for layer in layers:
        cfg = layer.get('config', {})
        if layer.get('class_name') == 'GlobalAveragePooling2D':
            # TF.js expects keepDims (camelCase)
            cfg['keepDims'] = True
            if 'keepdims' in cfg:
                del cfg['keepdims']
            
        if 'layers' in cfg:
            fix_keepdims(cfg['layers'])

fix_keepdims(layers)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(m, f)

print("Patched GlobalAveragePooling2D to have keepDims=True")
