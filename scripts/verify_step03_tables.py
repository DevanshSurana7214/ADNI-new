#!/usr/bin/env python3
"""Verify Step 3 S1 class tables independently from splits.json and episodes.csv."""

import json
import pandas as pd
from pathlib import Path

repo_root = Path(__file__).parent.parent

# Load splits.json
with open(repo_root / 'results' / 'step03' / 'splits.json', 'r') as f:
    splits = json.load(f)

# Load episodes.csv
episodes = pd.read_csv(repo_root / 'docs' / 'episodes.csv')

# y to DIAGNOSIS mapping
y_to_dx = {0: 'CN', 1: 'MCI', 2: 'Dementia'}

print("STEP 3 S1 TEST SET VERIFICATION\n" + "="*80 + "\n")

# Global cohort counts
total_counts = episodes['y'].value_counts().sort_index()
print("Overall cohort (all 13240 episodes):")
print(f"  CN:       {total_counts[0]:4d} ({100*total_counts[0]/len(episodes):.1f}%)")
print(f"  MCI:      {total_counts[1]:4d} ({100*total_counts[1]/len(episodes):.1f}%)")
print(f"  Dementia: {total_counts[2]:4d} ({100*total_counts[2]/len(episodes):.1f}%)")
print(f"  Total:    {len(episodes):4d}\n")

# Verify for each seed
for seed_idx in range(3):
    seed_key = f'seed_{seed_idx}'
    print(f"SEED {seed_idx}")
    print("-" * 80)

    seed_data = splits['s1'][seed_key]

    # Collect test episode IDs and counts across all 5 folds
    all_test_cn = 0
    all_test_mci = 0
    all_test_dem = 0
    all_test_count = 0

    for fold_idx in range(5):
        fold_key = str(fold_idx)
        fold_data = seed_data[fold_key]
        test_episode_ids = fold_data['episode_ids']['test']

        # Get diagnoses for test episodes
        test_episodes = episodes[episodes['episode_id'].isin(test_episode_ids)]
        test_counts = test_episodes['y'].value_counts().sort_index()

        cn_count = test_counts.get(0, 0)
        mci_count = test_counts.get(1, 0)
        dem_count = test_counts.get(2, 0)
        total = len(test_episode_ids)

        # Accumulate for seed totals
        all_test_cn += cn_count
        all_test_mci += mci_count
        all_test_dem += dem_count
        all_test_count += total

        # Compute percentages
        cn_pct = 100 * cn_count / total
        mci_pct = 100 * mci_count / total
        dem_pct = 100 * dem_count / total

        print(f"  Fold {fold_idx}: {total:4d} episodes")
        print(f"    CN:       {cn_count:4d} ({cn_pct:4.1f}%)")
        print(f"    MCI:      {mci_count:4d} ({mci_pct:4.1f}%)")
        print(f"    Dementia: {dem_count:4d} ({dem_pct:4.1f}%)")

    print(f"\n  SEED {seed_idx} TOTAL TEST (sum of 5 folds): {all_test_count} episodes")
    print(f"    CN:       {all_test_cn:4d}")
    print(f"    MCI:      {all_test_mci:4d}")
    print(f"    Dementia: {all_test_dem:4d}")

    # Assert totals match global counts
    assert all_test_cn == total_counts[0], f"Seed {seed_idx}: CN sum {all_test_cn} != global {total_counts[0]}"
    assert all_test_mci == total_counts[1], f"Seed {seed_idx}: MCI sum {all_test_mci} != global {total_counts[1]}"
    assert all_test_dem == total_counts[2], f"Seed {seed_idx}: Dementia sum {all_test_dem} != global {total_counts[2]}"
    print(f"    OK: Sums match global cohort counts (5263 / 5518 / 2459)\n")

print("\n" + "="*80)
print("OK: All assertions passed: five test folds per seed sum exactly to global counts.")
print("="*80 + "\n")
