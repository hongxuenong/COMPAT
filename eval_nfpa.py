"""
eval_nfpa.py — backward-compatible shim.

The evaluation is now unified in ``eval.py``. This runs it with the NFPA attack:

    python eval_nfpa.py            # equivalent to: python eval.py --attack nfpa

Prefer ``python eval.py --attack nfpa`` (see eval.py for all options).
"""

from eval import run_evaluation

if __name__ == "__main__":
    run_evaluation(attack="nfpa")
