#Run the thesis pipeline in stage order or selectively.

import subprocess
import sys
import os
import time
import argparse

STAGES = [
    (1, "create_thesis_input_files.py",
     "Build Large/Medium/Small filtered files",
     "~5 min"),
    (2, "calculate_traffic_metrics.py",
     "Compute traffic metrics on full BTS scope",
     "~10-15 min"),
    (3, "calculate_turnaround_metrics.py",
     "Compute turnaround metrics on full BTS scope",
     "~2-5 min"),
    (4, "build_unified_60.py",
     "Concatenate 3 filtered files into input_data_unified_60.csv",
     "~1 min"),
    (5, "thesis_main.py",
     "Compute ANSP delay scores and network features",
     "~30-60 min"),
    (6, "data_and_methods_outputs.py",
     "Generate Data and Methods figures (fig7, fig8)",
     "~1 min"),
    (7, "thesis_empirical_validation.py",
     "Hub-stratified model training and ANSP lift evaluation",
     "~3-5 hours"),
]

def parse_stages(spec):
    if not spec:
        return set(s[0] for s in STAGES)
    stages = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            stages.update(range(int(lo), int(hi) + 1))
        else:
            stages.add(int(part))
    return stages

def run_stage(stage_num, script, description, duration):
    print(f"\nStage {stage_num}: {script}  [{duration}]")
    print(f"  {description}")

    python_exe = os.path.join('.venv_win', 'Scripts', 'python.exe')
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    t0 = time.time()
    result = subprocess.run([python_exe, '-u', script])
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[FAIL] Stage {stage_num} ({script}) failed after {elapsed:.0f}s")
        sys.exit(1)
    print(f"\n[OK] Stage {stage_num} completed in {elapsed:.0f}s")

def main():
    parser = argparse.ArgumentParser(description="Run the thesis data pipeline.")
    parser.add_argument('--stages', default=None,
                        help='Stages to run, e.g. "1-4", "6", "2,4,5". Default: all.')
    parser.add_argument('--from', dest='from_stage', type=int, default=None,
                        help='Run from this stage to the end.')
    parser.add_argument('--list', action='store_true',
                        help='List all stages and exit.')
    args = parser.parse_args()

    if args.list:
        for num, script, desc, dur in STAGES:
            print(f"  {num}. {script:<40} {desc} ({dur})")
        return

    if args.from_stage is not None:
        selected = set(range(args.from_stage, len(STAGES) + 1))
    else:
        selected = parse_stages(args.stages)

    print(f"\nRunning stages: {sorted(selected)}")
    for num, script, desc, dur in STAGES:
        marker = ">>" if num in selected else "  "
        print(f"  {marker} {num}. {script:<40} ({dur})")

    overall_start = time.time()
    for num, script, desc, dur in STAGES:
        if num not in selected:
            continue
        if not os.path.exists(script):
            print(f"\n[FAIL] Missing script: {script}")
            sys.exit(1)
        run_stage(num, script, desc, dur)

    total = time.time() - overall_start
    print(f"\nDone. Total time: {total:.0f}s ({total/60:.1f} min)")

if __name__ == '__main__':
    main()
