import json

path = 'public/model/model.json'
with open(path, 'r', encoding='utf-8') as f:
    m = json.load(f)

def fix_dtype(obj):
    if isinstance(obj, dict):
        if 'dtype' in obj and isinstance(obj['dtype'], dict):
            # Try to extract the name from DTypePolicy
            policy = obj['dtype']
            if policy.get('class_name') == 'DTypePolicy':
                obj['dtype'] = policy.get('config', {}).get('name', 'float32')
            else:
                # Fallback to float32 if we can't figure it out
                obj['dtype'] = 'float32'
        
        # Also fix any other nested dicts
        for k, v in obj.items():
            fix_dtype(v)
    elif isinstance(obj, list):
        for item in obj:
            fix_dtype(item)

# Start fixing from the top config
if 'modelTopology' in m and 'config' in m['modelTopology']:
    fix_dtype(m['modelTopology']['config'])

with open(path, 'w', encoding='utf-8') as f:
    json.dump(m, f)
print("Converted all complex dtype objects to simple strings.")
