#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

CATEGORY_MAP = {
    'Math': 'STEM',
    'Biology': 'STEM',
    'Chemistry': 'STEM',
    'Computer-Science': 'STEM',
    'Electrical-Engineering': 'STEM',
    'Electronics-Engineering': 'STEM',
    'Mechanical-Engineering': 'STEM',
    'Chemical-Engineering': 'STEM',
    'Materials-Engineering': 'STEM',
    'Civil-Engineering': 'STEM',
    'Environmental-Science': 'STEM',
    'Ecology': 'STEM',
    'Agricultural-Sciences': 'Applied Science',
    'Food-Processing': 'Applied Science',
    'Health': 'Applied Science',
    'Industrial-Engineer': 'Applied Science',
    'Machine-Design-and-Manufacturing': 'Applied Science',
    'Construction': 'Applied Science',
    'Energy-Management': 'Applied Science',
    'Gas-Technology-and-Engineering': 'Applied Science',
    'Geomatics': 'Applied Science',
    'Maritime-Engineering': 'Applied Science',
    'Railway-and-Automotive-Engineering': 'Applied Science',
    'Aviation-Engineering-and-Maintenance': 'Applied Science',
    'Nondestructive-Testing': 'Applied Science',
    'Refrigerating-Machinery': 'Applied Science',
    'Telecommunications-and-Wireless-Technology': 'Applied Science',
    'Korean-History': 'HUMSS',
    'Economics': 'HUMSS',
    'Political-Science-and-Sociology': 'HUMSS',
    'Psychology': 'HUMSS',
    'Education': 'HUMSS',
    'Law': 'Other',
    'Criminal-Law': 'Other',
    'Management': 'Other',
    'Marketing': 'Other',
    'Accounting': 'Other',
    'Taxation': 'Other',
    'Patent': 'Other',
    'Real-Estate': 'Other',
    'Social-Welfare': 'Other',
    'Public-Safety': 'Other',
    'Information-Technology': 'Other',
    'Interior-Architecture-and-Design': 'Other',
    'Fashion': 'Other'
}

with open('my_experiments/kmmlu_ax_4.0_light_sft.json') as f:
    sft = json.load(f)

category_stats = {
    'STEM': {'correct': 0, 'total': 0},
    'Applied Science': {'correct': 0, 'total': 0},
    'HUMSS': {'correct': 0, 'total': 0},
    'Other': {'correct': 0, 'total': 0}
}

for subject, data in sft.items():
    category = CATEGORY_MAP.get(subject, 'Other')
    category_stats[category]['correct'] += data['correct']
    category_stats[category]['total'] += data['total']

category_accuracy = {}
for cat, stats in category_stats.items():
    if stats['total'] > 0:
        accuracy = (stats['correct'] / stats['total']) * 100
        category_accuracy[cat] = accuracy

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("📊 SFT 카테고리별 정확도")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

for cat in ['STEM', 'Applied Science', 'HUMSS', 'Other']:
    stats = category_stats[cat]
    acc = category_accuracy.get(cat, 0)
    print(f"{cat:20s}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("📈 Zero-shot vs SFT 카테고리 비교")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

zs_categories = {
    'Applied Science': 53.58,
    'HUMSS': 58.75,
    'Other': 56.64,
    'STEM': 55.32
}

print(f"{'카테고리':<20s} {'Zero-shot':>12s} {'SFT':>12s} {'향상':>12s}")
print("─" * 60)
for cat in ['STEM', 'Applied Science', 'HUMSS', 'Other']:
    zs_acc = zs_categories[cat]
    sft_acc = category_accuracy.get(cat, 0)
    improvement = sft_acc - zs_acc
    print(f"{cat:<20s} {zs_acc:>11.2f}% {sft_acc:>11.2f}% {improvement:>+11.2f}%p")

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
