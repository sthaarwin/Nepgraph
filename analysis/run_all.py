import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(ROOT_DIR, "analysis")

scripts = [
    "k_sweep.py",
    "null_model.py",
    "stability.py",
    "centrality_comparison.py",
    "anomaly_definition.py",
    "portfolio_backtest.py"
]

def main():
    print("Starting full reproducible run of NepGraph Analysis Pipeline...\n")
    for script in scripts:
        script_path = os.path.join(ANALYSIS_DIR, script)
        print(f"{'='*50}")
        print(f"Running {script}...")
        print(f"{'='*50}")
        try:
            subprocess.run([sys.executable, script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}. Exiting.")
            sys.exit(1)
    print("\nAll analyses completed successfully. Figures and results are up to date.")

if __name__ == "__main__":
    main()
