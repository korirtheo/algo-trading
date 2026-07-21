"""
G+L Trail Optimization — thin shim.
The full implementation lives in optimize_combined.py (--study-type gl_trail).
Run via: python optimize_combined.py --study-type gl_trail [args]
Or:      ./run_optuna_gl_trail.sh
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from optimize_combined import main
if __name__ == '__main__':
    sys.argv.insert(1, '--study-type')
    sys.argv.insert(2, 'gl_trail')
    main()
