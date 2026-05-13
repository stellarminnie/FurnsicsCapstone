"""
Fix all TFjs-incompatible config fields in model.json:

1. dtype: Keras 3 stores dtype as a DTypePolicy object dict.
   TFjs expects a plain string like "float32".
   Fix: replace any dict dtype with its config.name string.

2. keepDims -> keepdims: GlobalAveragePooling2D uses camelCase in Keras 3.
   TFjs expects lowercase 'keepdims'.

3. activation: If activation is stored as a dict (advanced Keras 3 format)
   with class_name/config, simplify to just the string activation name.

Operates in-place on public/model/model.json.
"""
import json

MODEL_PATH = r'public\model\model.json'

with open(MODEL_PATH, 'r', encoding='utf-8') as f:
    m = json.load(f)

fixed_dtype = 0
fixed_keepdims = 0
fixed_activation = 0

def fix_layer_config(cfg, layer_name=''):
    global fixed_dtype, fixed_keepdims, fixed_activation

    # 1. Fix dtype: dict -> string
    if isinstance(cfg.get('dtype'), dict):
        policy = cfg['dtype']
        # DTypePolicy has config.name
        dtype_str = policy.get('config', {}).get('name', 'float32')
        cfg['dtype'] = dtype_str
        fixed_dtype += 1

    # 2. Fix keepDims -> keepdims
    if 'keepDims' in cfg:
        cfg['keepdims'] = cfg.pop('keepDims')
        fixed_keepdims += 1

    # 3. Fix activation stored as dict
    act = cfg.get('activation')
    if isinstance(act, dict):
        # Could be {"class_name": "relu", "config": {}} or
        # {"module": ..., "class_name": "Linear", "config": {}, "registered_name": null}
        act_name = act.get('class_name', 'linear')
        if act_name == 'Linear':
            act_name = 'linear'
        cfg['activation'] = act_name
        fixed_activation += 1

def walk_layers(layers):
    for l in layers:
        fix_layer_config(l.get('config', {}), l.get('name', ''))
        # Recurse into nested model configs (just in case)
        inner = l.get('config', {}).get('layers')
        if inner:
            walk_layers(inner)

# Walk all layers
top = m['modelTopology']
root_cfg = top.get('model_config', top)['config']
walk_layers(root_cfg.get('layers', []))

with open(MODEL_PATH, 'w', encoding='utf-8') as f:
    json.dump(m, f)

print(f"Fixed {fixed_dtype} dtype fields (dict -> string)")
print(f"Fixed {fixed_keepdims} keepDims -> keepdims")
print(f"Fixed {fixed_activation} activation fields (dict -> string)")
print("model.json saved.")
