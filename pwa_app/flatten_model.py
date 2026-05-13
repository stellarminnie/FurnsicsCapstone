"""
Flatten the nested Functional (MobileNetV3Small) sub-model into the outer model.

TFjs loadLayersModel cannot handle a Keras 3 exported model where one of the
layers is itself a Functional sub-model.  The fix is to "inline" all inner
layers into the outer layers list and fix up inbound_nodes references so that:

  Before:
    input_layer_5 → Functional(MobileNetV3Small) → global_average_pooling2d_1 → …

  After:
    input_layer_5 → conv → … → activation_35 → global_average_pooling2d_1 → …

Steps:
  1. Extract MobileNetV3Small's inner layers list.
  2. Remove its own InputLayer (input_layer_4) — we already have input_layer_5.
  3. Fix every inner layer that referenced input_layer_4 to reference input_layer_5.
  4. Remove the Functional wrapper layer from the outer list.
  5. Inject all surviving inner layers into the outer list (right after input_layer_5).
  6. Fix global_average_pooling2d_1 (and any other outer layer) that referenced
     'MobileNetV3Small' to instead reference 'activation_35' (the inner output).
  7. Fix outer model's input_layers / output_layers if present.
"""

import json, copy

MODEL_PATH = 'public/model/model.json'

with open(MODEL_PATH, 'r', encoding='utf-8') as f:
    m = json.load(f)

top = m['modelTopology']
root_cfg = top.get('model_config', top)['config']
outer_layers = root_cfg['layers']

# ── 1. Locate the Functional sub-model layer ──────────────────────────────────
mobilenet_layer = None
mobilenet_idx   = None
for i, l in enumerate(outer_layers):
    if l.get('class_name') == 'Functional' and l.get('name') == 'MobileNetV3Small':
        mobilenet_layer = l
        mobilenet_idx   = i
        break

if mobilenet_layer is None:
    print("No nested MobileNetV3Small Functional found — nothing to do.")
    exit(0)

inner_layers = mobilenet_layer['config']['layers']
inner_input_name  = None   # the InputLayer inside MobileNetV3Small
inner_output_name = None   # the final layer name inside MobileNetV3Small

# Detect inner input / output names from the sub-model config
inner_input_layers  = mobilenet_layer['config'].get('input_layers')
inner_output_layers = mobilenet_layer['config'].get('output_layers')

if isinstance(inner_input_layers, list):
    if isinstance(inner_input_layers[0], list):
        inner_input_name = inner_input_layers[0][0]   # [["input_layer_4",0,0]]
    else:
        inner_input_name = inner_input_layers[0]       # ["input_layer_4",0,0]

if isinstance(inner_output_layers, list):
    if isinstance(inner_output_layers[0], list):
        inner_output_name = inner_output_layers[0][0]
    else:
        inner_output_name = inner_output_layers[0]

print(f"Inner model input  layer: {inner_input_name}")
print(f"Inner model output layer: {inner_output_name}")

# ── 2. Determine what input_layer_5 connects to ──────────────────────────────
# The outer model feeds input_layer_5 into MobileNetV3Small.
# MobileNetV3Small internally uses input_layer_4.
# We need to rewire every inner layer that pulls from input_layer_4
# to pull from input_layer_5 instead.
outer_input_name = None
for l in outer_layers:
    if l.get('class_name') == 'InputLayer':
        outer_input_name = l['name']
        break
print(f"Outer model input  layer: {outer_input_name}")

# ── 3. Deep-copy inner layers; remove their InputLayer; fix inbound refs ──────
surviving_inner = []
for l in inner_layers:
    if l.get('class_name') == 'InputLayer' and l.get('name') == inner_input_name:
        # Drop the inner InputLayer entirely
        continue
    lc = copy.deepcopy(l)
    # Replace any reference to inner_input_name with outer_input_name
    def fix_inbound(nodes):
        if not nodes:
            return nodes
        result = []
        for node in nodes:
            if isinstance(node, list):
                fixed = []
                for ref in node:
                    if isinstance(ref, list) and ref[0] == inner_input_name:
                        ref = [outer_input_name] + ref[1:]
                    fixed.append(ref)
                result.append(fixed)
            else:
                result.append(node)
        return result

    lc['inbound_nodes'] = fix_inbound(lc.get('inbound_nodes', []))
    surviving_inner.append(lc)

print(f"Inlining {len(surviving_inner)} inner layers (dropped InputLayer '{inner_input_name}')")

# ── 4. Build the new outer layers list ───────────────────────────────────────
# Insert inner layers right after input_layer_5 (position mobilenet_idx).
# Remove the Functional wrapper at mobilenet_idx.
new_outer = (
    outer_layers[:mobilenet_idx]        # everything before MobileNetV3Small
    + surviving_inner                    # inlined inner layers
    + outer_layers[mobilenet_idx + 1:]  # everything after MobileNetV3Small
)

# ── 5. Fix any outer layer that referenced 'MobileNetV3Small' ────────────────
def fix_mobilenet_ref(nodes, old_name, new_name):
    if not nodes:
        return nodes
    result = []
    for node in nodes:
        if isinstance(node, list):
            fixed = []
            for ref in node:
                if isinstance(ref, list) and ref[0] == old_name:
                    ref = [new_name] + ref[1:]
                fixed.append(ref)
            result.append(fixed)
        else:
            result.append(node)
    return result

for l in new_outer:
    l['inbound_nodes'] = fix_mobilenet_ref(
        l.get('inbound_nodes', []), 'MobileNetV3Small', inner_output_name
    )

root_cfg['layers'] = new_outer

# Also patch top-level input_layers / output_layers if they reference the submodel
for key in ('input_layers', 'output_layers'):
    val = root_cfg.get(key)
    if isinstance(val, list):
        if isinstance(val[0], list) and val[0][0] == 'MobileNetV3Small':
            root_cfg[key] = [[inner_output_name] + val[0][1:]]

# ── 6. Write back ─────────────────────────────────────────────────────────────
with open(MODEL_PATH, 'w', encoding='utf-8') as f:
    json.dump(m, f)

print(f"\n✓ Flattened model written to {MODEL_PATH}")
print(f"  Total outer layers: {len(new_outer)}")
