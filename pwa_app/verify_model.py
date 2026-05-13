import json

with open(r'public\model\model.json', 'r', encoding='utf-8') as f:
    m = json.load(f)

top = m['modelTopology']
if 'model_config' in top:
    layers = top['model_config']['config']['layers']
else:
    layers = top['config']['layers']

def find_input_layers(layers):
    for l in layers:
        cfg = l.get('config', {})
        if l.get('class_name') == 'InputLayer':
            print(f"  InputLayer '{cfg.get('name')}': batchInputShape={cfg.get('batchInputShape')}, inputShape={cfg.get('inputShape')}")
        if 'layers' in cfg:
            find_input_layers(cfg['layers'])

find_input_layers(layers)
