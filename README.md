# Simulation code for "Prescribed performance nonlinear model predictive control of fed-batch polyhydroxyalkanoate production with life cycle and techno-economic assessment"

Requirements: Python 3 with numpy, scipy, pandas, openpyxl, scikit-learn, casadi (with IPOPT) and matplotlib.
Input data: master.csv, the twelve literature datasets (DS01 to DS12) with columns id, t, S0, N0, S, N, X, P (time in h, concentrations in g/L; N as ammonium sulphate, converted to nitrogen in calib2.py).

Run order
1. ident_run.py      identifiability of candidate free parameter sets (Table S2) and run screening (Table S3)
2. final_calib.py    published-parameter test and leave-one-run-out test on DS01, DS08, DS11, DS12 (Table 1 of the main text, Table S4),
                     reference experiment (Table S5), plant parameter vectors (Table S7); writes calib_reduced.json
3. ann.py            training data and soft sensor (Table S10); writes ann7.pkl
4. experiments2.py   closed-loop runs: nominal, parametric mismatch, soft sensor, literature maximum PHA ratio
                     (Table 4 of the main text, Tables S8, S9, S11, S12, Section S8). Arguments: nitrogen feed bound, tag, mode
5. figs2.py          Figures 4 (main text) and S1 to S4

Modules: model.py (kinetic model and parameter set of Table S13), nmpc.py (PP-NMPC and baseline NMPC), refexp.py (reference experiment data
and reconstruction), calib2.py (calibration utilities), ident.py (confidence intervals), summ.py (summary statistics).
