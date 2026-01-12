import torch

config = {
    'device': 'cuda',
    'batch_size': 16,
    'lang_emb_dim': 768,
    'lang_num_layers': 4,
    'lang_lr': 1e-4,
    'lang_weight_decay': 1e-2,
    'contrastive_temperature': 0.07,
    'contrastive_weight': 1.0,
    'l2_weight': 0.1,
}
