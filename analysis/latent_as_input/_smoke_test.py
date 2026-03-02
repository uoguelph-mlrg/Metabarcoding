import sys, os, logging

# Local folder must come before src so we import the local train.py
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
# src is appended (lower priority) — train.py will pick it up via its own sys.path.append
sys.path.append(os.path.join(_here, '..', '..', 'src'))

logging.basicConfig(level=logging.WARNING)

from config import Config, set_seed
from train import Trainer

set_seed(42)
cfg = Config()
cfg.epochs = 3
cfg.patience = None

t = Trainer(cfg, data_dir='../../data', loss_type='cross_entropy')
res = t.run(use_wandb=False)
Z = res['latent_embeddings']
print(f"OK — best_val={res['best_val_loss']:.6f}, latent std={Z.std():.6f}, latent norm={Z.sum():.4f}")
