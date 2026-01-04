#!/usr/bin/env python3
"""
Create config.pkl from base_config.py for evaluation.
Place this in the checkpoints directory alongside model.pt
"""
import sys
import os
sys.path.insert(0, './instant_policy')

from ip.configs.base_config import config
import pickle

# Modify config to disable pre-trained encoder since scene_encoder.pt is not available
config['pre_trained_encoder'] = False
config['freeze_encoder'] = False

# Save config to checkpoints directory
user = os.environ.get('USER', 'kanth042')
output_path = f'/scratch.global/{user}/ips/checkpoints/config.pkl'

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'wb') as f:
    pickle.dump(config, f)

print(f"Created {output_path}")
print("\nConfig contents:")
for key, value in config.items():
    print(f"  {key}: {value}")
