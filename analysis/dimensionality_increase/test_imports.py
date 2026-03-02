#!/usr/bin/env python3
"""Test that we're importing the correct modules."""

import sys
from pathlib import Path

root_dir = Path(__file__).parent
src_path = str(root_dir.parent.parent / "src")

print("=== BASELINE IMPORTS ===")
sys.path.insert(0, src_path)
import config as baseline_config
print(f"baseline_config file: {baseline_config.__file__}")
print(f"baseline_config has embed_dim field: {hasattr(baseline_config.Config, '__dataclass_fields__') and 'embed_dim' in baseline_config.Config.__dataclass_fields__}")
# Check if embed_dim is in the Config class annotations
import inspect
baseline_sig = inspect.signature(baseline_config.Config.__init__)
print(f"baseline_config.__init__ params: {list(baseline_sig.parameters.keys())}")
sys.path.pop(0)

# Clear modules to avoid caching
for mod in list(sys.modules.keys()):
    if mod in ['config', 'train', 'model', 'latent_solver', 'mlp']:
        del sys.modules[mod]

print("\n=== LOCAL IMPORTS ===")
sys.path.insert(0, str(root_dir))
import config as local_config_module
print(f"local_config file: {local_config_module.__file__}")
print(f"local_config has embed_dim field: {hasattr(local_config_module.Config, '__dataclass_fields__') and 'embed_dim' in local_config_module.Config.__dataclass_fields__}")
local_sig = inspect.signature(local_config_module.Config.__init__)
print(f"local_config.__init__ params: {list(local_sig.parameters.keys())}")

# Try to create instance
try:
    cfg = local_config_module.Config(embed_dim=8, gating_fn='sigmoid')
    print(f"✅ local_config created with embed_dim={cfg.embed_dim}, gating_fn={cfg.gating_fn}")
except Exception as e:
    print(f"❌ Error creating local_config: {e}")

sys.path.pop(0)

print("\n=== FILE COMPARISON ===")
print(f"baseline_config from: {baseline_config.__file__}")
print(f"local_config from: {local_config_module.__file__}")
print(f"Same file? {baseline_config.__file__ == local_config_module.__file__}")
