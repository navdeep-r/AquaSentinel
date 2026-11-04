"""Launcher script for running AquaSentinel Dashboard directly via:
    python -m src.dashboard
"""

import os
import subprocess
import sys


def main():
    dashboard_app_path = os.path.join(os.path.dirname(__file__), "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_app_path]
    print(f"Launching AquaSentinel Mission Control Dashboard: {' '.join(cmd)}")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nAquaSentinel Dashboard stopped.")


if __name__ == "__main__":
    main()
