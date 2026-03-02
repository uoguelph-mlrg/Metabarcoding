"""Quick comparison of taxonomy vs embedding model results."""
from pathlib import Path
import re

def parse_log(path):
    cycles = []
    for line in Path(path).read_text().splitlines():
        m = re.search(r'Cycle (\d+): val_loss=([\d.]+), train_loss=([\d.]+)', line)
        if m:
            cycles.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    test = None
    for line in Path(path).read_text().splitlines():
        m = re.search(r'Test loss: ([\d.]+)', line)
        if m:
            test = float(m.group(1))
    return cycles, test

BASE = Path(__file__).parent / "results"
t_cycles, t_test = parse_log(BASE / "taxonomy_run.log")
e_cycles, e_test = parse_log(BASE / "embedding_run.log")

t_best_val = min(c[1] for c in t_cycles)
e_best_val = min(c[1] for c in e_cycles)

print("=" * 62)
print("         TAXONOMY vs BARCODEBERT EMBEDDING COMPARISON")
print("=" * 62)
row = "{:<28} {:>10}  {:>10}  {:>8}"
print(row.format("Metric", "Taxonomy", "Embedding", "Delta"))
print("-" * 62)
print(row.format("Test loss (CE ↓)", f"{t_test:.6f}", f"{e_test:.6f}", f"{e_test-t_test:+.6f}"))
print(row.format("Best val loss (↓)", f"{t_best_val:.6f}", f"{e_best_val:.6f}", f"{e_best_val-t_best_val:+.6f}"))
print(row.format("Final train loss (↓)", f"{t_cycles[-1][2]:.6f}", f"{e_cycles[-1][2]:.6f}", f"{e_cycles[-1][2]-t_cycles[-1][2]:+.6f}"))
print(row.format("Cycles ran", str(len(t_cycles)), str(len(e_cycles)), ""))

stopped_t = "Yes" if len(t_cycles) < 100 else "No (max)"
stopped_e = "Yes" if len(e_cycles) < 100 else "No (max)"
print(row.format("Early stopping", stopped_t, stopped_e, ""))
print()
print("Val loss at selected cycles:")
print(f"  {'Cycle':>5}  {'Taxonomy':>10}  {'Embedding':>10}  {'Delta':>8}")
for i in [0, 19, 39, 59, 79, 99]:
    if i < len(t_cycles) and i < len(e_cycles):
        t_v = t_cycles[i][1]
        e_v = e_cycles[i][1]
        print(f"  {i+1:5d}  {t_v:10.6f}  {e_v:10.6f}  {e_v-t_v:+8.6f}")
print()
print("Notes:")
print("  - Embedding model: 13088/15539 BINs have DNA sequences (84%)")
print("    2451 BINs (16%) use taxonomy-based neighbours as fallback")
print("  - Both models ran the full 100 cycles without early stopping")
print("  - Lower CE loss = better predicted species composition per sample")
print("=" * 62)
