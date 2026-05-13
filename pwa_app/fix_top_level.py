import json
path = 'public/model/model.json'
with open(path, 'r', encoding='utf-8') as f:
    m = json.load(f)

top = m.get('modelTopology', {})
if 'model_config' in top and 'class_name' not in top:
    mc = top['model_config']
    top['class_name'] = mc.get('class_name', 'Functional')
    top['config'] = mc.get('config', {})
    del top['model_config']
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(m, f)
    print("Fixed modelTopology structure for TF.js (promoted model_config)")
else:
    print("Already fixed or model_config not found.")
