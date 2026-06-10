#!/usr/bin/env python3
"""
Monitor word count compliance in real-time as the attack runs.
Usage: python monitor_compliance.py [architecture] [interval_seconds]
Example: python monitor_compliance.py llama3.3 30
"""
import csv
import glob
import os
import sys
import time

def get_latest_file(arch):
    files = sorted(glob.glob(f'results/{arch}_*.csv'), key=os.path.getmtime, reverse=True)
    return files[0] if files else None

def check_compliance(arch, interval=30):
    prev_rows = 0
    
    while True:
        latest = get_latest_file(arch)
        if not latest:
            print(f"Waiting for {arch} result file...")
            time.sleep(interval)
            continue
        
        total = 0
        in_range = 0
        row_count = 0
        row_stats = {}
        all_over_words = []
        all_under_words = []
        
        try:
            with open(latest, newline='') as fh:
                for r_i, row in enumerate(csv.DictReader(fh), 1):
                    row_count = r_i
                    orig = len((row.get('original_text') or '').split())
                    temp = float(row.get('temperature', 0.0))
                    lo, hi = orig - 20, orig + 20
                    row_ok = 0
                    row_total = 0
                    row_over_words = []
                    row_under_words = []
                    
                    all_wcs = []
                    for col in [c for c in row if c.startswith('rewrite') and c.endswith('_text')]:
                        text = (row.get(col) or '').strip()
                        if not text:
                            continue
                        wc = len(text.split())
                        ok = lo <= wc <= hi
                        total += 1
                        in_range += int(ok)
                        row_total += 1
                        row_ok += int(ok)
                        all_wcs.append(wc)
                        
                        if wc > hi:
                            over_delta = wc - hi
                            all_over_words.append(over_delta)
                            row_over_words.append(over_delta)
                        elif wc < lo:
                            under_delta = lo - wc
                            all_under_words.append(under_delta)
                            row_under_words.append(under_delta)
                    
                    if row_total > 0:
                        avg_wc = sum(all_wcs) / len(all_wcs)
                        row_over_count = len(row_over_words)
                        row_under_count = len(row_under_words)
                        row_avg_over = sum(row_over_words) / row_over_count if row_over_count else 0
                        row_avg_under = sum(row_under_words) / row_under_count if row_under_count else 0
                        row_stats[r_i] = (
                            row_ok,
                            row_total,
                            orig,
                            temp,
                            avg_wc,
                            row_over_count,
                            row_under_count,
                            row_avg_over,
                            row_avg_under,
                        )
        except:
            time.sleep(interval)
            continue
        
        if row_count > prev_rows:
            overall_avg_over = sum(all_over_words) / len(all_over_words) if all_over_words else 0
            overall_avg_under = sum(all_under_words) / len(all_under_words) if all_under_words else 0
            overall_pct = (100 * in_range / total) if total else 0.0
            over_str = f"over:{len(all_over_words)}(avg +{overall_avg_over:.1f}w)" if all_over_words else "over:0"
            under_str = f"under:{len(all_under_words)}(avg -{overall_avg_under:.1f}w)" if all_under_words else "under:0"

            for r_i in range(prev_rows + 1, row_count + 1):
                if r_i in row_stats:
                    (
                        row_ok,
                        row_total,
                        orig,
                        temp,
                        avg_wc,
                        row_over_count,
                        row_under_count,
                        row_avg_over,
                        row_avg_under,
                    ) = row_stats[r_i]
                    row_pct = (100 * row_ok / row_total) if row_total else 0.0
                    row_over_str = f"over:{row_over_count}(avg +{row_avg_over:.1f}w)" if row_over_count else "over:0"
                    row_under_str = f"under:{row_under_count}(avg -{row_avg_under:.1f}w)" if row_under_count else "under:0"
                    print(f"[{time.strftime('%H:%M:%S')}] Row {r_i:3d} (orig={orig}, temp={temp:.2f}): {row_ok}/{row_total} ({row_pct:.0f}%) | avg_wc={avg_wc:.1f} | {over_str} {under_str} | Overall: {in_range}/{total} ({overall_pct:.1f}%)")
                    print(f"    Row stats: {row_over_str} {row_under_str}")
            prev_rows = row_count
        
        time.sleep(interval)

if __name__ == '__main__':
    arch = sys.argv[1] if len(sys.argv) > 1 else 'llama3.3'
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print(f"Monitoring {arch} with {interval}s interval...\n")
    check_compliance(arch, interval)
