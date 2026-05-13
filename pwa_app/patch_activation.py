import json

path = 'public/model/model.json'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace hard_silu with swish (which is supported by tfjs)
new_text = text.replace('"hard_silu"', '"swish"')

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_text)

print("Replaced hard_silu with swish in model.json")
