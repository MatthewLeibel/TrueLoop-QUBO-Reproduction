#!/usr/bin/env bash
set -e
python3 exp1_feedback_channel.py
python3 exp2_structured_solve.py
python3 exp3_boundary_rows.py
echo "Done. Compare results/*.json to README tables."
