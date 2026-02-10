"""Verify correlation between dataset distribution and model performance."""
import pandas as pd
import sys

# Map species codes to full names
species_mapping = {
    'barbar': 'Barbastellus barbastellus',
    'eptser': 'Eptesicus serotinus',
    'myoalc': 'Myotis alcathoe',
    'myobec': 'Myotis bechsteinii',
    'myobra': 'Myotis brandtii',
    'myodau': 'Myotis daubentonii',
    'myomys': 'Myotis mystacinus',
    'myonat': 'Myotis nattereri',
    'nyclei': 'Nyctalus leisleri',
    'nycnoc': 'Nyctalus noctula',
    'pipnat': 'Pipistrellus nathusii',
    'pippip': 'Pipistrellus pipistrellus',
    'pippyg': 'Pipistrellus pygmaeus',
    'pleaur': 'Plecotus auritus',
    'pleaus': 'Plecotus austriacus',
    'rhifer': 'Rhinolophus ferrumequinum',
    'rhihip': 'Rhinolophus hipposideros',
}

# Dataset distribution (from check_dataset_distribution.py output)
dataset_stats = {
    'Barbastellus barbastellus': {'train': 468, 'val': 577, 'ratio': 0.81},
    'Eptesicus serotinus': {'train': 403, 'val': 2188, 'ratio': 0.18},
    'Myotis alcathoe': {'train': 374, 'val': 504, 'ratio': 0.74},
    'Myotis bechsteinii': {'train': 241, 'val': 637, 'ratio': 0.38},
    'Myotis brandtii': {'train': 352, 'val': 1609, 'ratio': 0.22},
    'Myotis daubentonii': {'train': 4058, 'val': 2373, 'ratio': 1.71},
    'Myotis mystacinus': {'train': 1381, 'val': 1439, 'ratio': 0.96},
    'Myotis nattereri': {'train': 2610, 'val': 102, 'ratio': 25.59},
    'Nyctalus leisleri': {'train': 696, 'val': 449, 'ratio': 1.55},
    'Nyctalus noctula': {'train': 209, 'val': 204, 'ratio': 1.02},
    'Pipistrellus nathusii': {'train': 1490, 'val': 0, 'ratio': float('inf')},
    'Pipistrellus pipistrellus': {'train': 877, 'val': 1038, 'ratio': 0.84},
    'Pipistrellus pygmaeus': {'train': 1496, 'val': 1132, 'ratio': 1.32},
    'Plecotus auritus': {'train': 528, 'val': 582, 'ratio': 0.91},
    'Plecotus austriacus': {'train': 331, 'val': 538, 'ratio': 0.62},
    'Rhinolophus ferrumequinum': {'train': 717, 'val': 1488, 'ratio': 0.48},
    'Rhinolophus hipposideros': {'train': 722, 'val': 1571, 'ratio': 0.46},
}

# Read latest metrics
df = pd.read_csv('outputs/logs/version_54/metrics.csv')
val_metrics = df[df['total_loss/val'].notna()].copy()
latest_epoch = val_metrics.iloc[-1]

print(f"Analysis for Epoch {int(latest_epoch['epoch'])}")
print("=" * 100)
print(f"{'Species':<30} {'AP %':<10} {'Train':<10} {'Val':<10} {'T/V Ratio':<12} {'Notes':<30}")
print("-" * 100)

results = []
for code, full_name in sorted(species_mapping.items()):
    col = f'classification/average_precision/{code}'
    ap = latest_epoch[col]
    
    if pd.isna(ap):
        ap_pct = 'NaN'
        ap_val = -1
    else:
        ap_pct = f"{ap * 100:.1f}%"
        ap_val = ap * 100
    
    stats = dataset_stats.get(full_name, {})
    train_count = stats.get('train', 0)
    val_count = stats.get('val', 0)
    ratio = stats.get('ratio', 0)
    
    # Determine issues
    notes = []
    if val_count == 0:
        notes.append("MISSING IN VAL")
    elif train_count < 300:
        notes.append("Low train count")
    if 0 < ratio < 0.5:
        notes.append("Underrep in train")
    elif ratio > 5:
        notes.append("Overrep in train")
    
    results.append({
        'species': full_name,
        'code': code,
        'ap': ap_val,
        'ap_str': ap_pct,
        'train': train_count,
        'val': val_count,
        'ratio': ratio,
        'notes': ', '.join(notes) if notes else ''
    })
    
    print(f"{full_name:<30} {ap_pct:<10} {train_count:<10} {val_count:<10} {ratio:<12.2f} {', '.join(notes):<30}")

print("-" * 100)

# Analysis
print("\n" + "=" * 100)
print("CORRELATION ANALYSIS")
print("=" * 100)

# Poor performers (< 50% AP)
poor = [r for r in results if 0 <= r['ap'] < 50]
print(f"\nPoor Performers (AP < 50%):")
for r in sorted(poor, key=lambda x: x['ap']):
    print(f"  {r['species']:<30} AP: {r['ap_str']:<10} Train: {r['train']:<6} Val: {r['val']:<6} Ratio: {r['ratio']:.2f}")
    if r['train'] < 300:
        print(f"    → Low training data ({r['train']} events)")
    if 0 < r['ratio'] < 0.5:
        print(f"    → Severe train/val imbalance (ratio {r['ratio']:.2f})")
    if r['val'] > r['train'] * 2:
        print(f"    → Val set much larger than train ({r['val']} vs {r['train']})")

# Good performers (> 80% AP)
good = [r for r in results if r['ap'] >= 80]
print(f"\nGood Performers (AP >= 80%):")
for r in sorted(good, key=lambda x: x['ap'], reverse=True):
    print(f"  {r['species']:<30} AP: {r['ap_str']:<10} Train: {r['train']:<6} Val: {r['val']:<6} Ratio: {r['ratio']:.2f}")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"Species with low training data (< 300 events): {len([r for r in results if r['train'] < 300])}")
print(f"Species with severe imbalance (ratio < 0.5 or > 5): {len([r for r in results if 0 < r['ratio'] < 0.5 or r['ratio'] > 5])}")
print(f"Species missing from validation: {len([r for r in results if r['val'] == 0])}")
