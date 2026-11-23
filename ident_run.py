import json, numpy as np
from calib2 import *
from ident import fim_ci
CAL=['DS01','DS11','DS12']; out=[]
for free in (['mu_xs','mu_ps'],['mu_xs','mu_ps','KPIN'],['mu_xs','mu_ps','KPIN','KS'],['mu_xs','mu_ps','KPIN','KS','KPS'],['mu_xs','mu_ps','KPIN','fmax','KS','KPS']):
    p,J=fit(CAL,free,maxiter=200); ci,corr,n,cond=fim_ci(p,free,CAL)
    out.append(dict(free=free,J=J,est=list(map(float,p)),ci_pct=list(map(float,100*ci/np.abs(p))),maxcorr=float(np.max(np.abs(corr-np.eye(len(p))))))); print(out[-1],flush=True)
# carbon balance with the fixed yields
sel={}
for ds in ['DS01','DS02','DS03','DS04','DS07','DS08','DS09','DS11','DS12']:
    d=run_data(ds); dS=d['S'][0]-d['S'][-1]; Xr=(d['X'][-1]-d['P'][-1])-(d['X'][0]-d['P'][0]); P=d['P'][-1]-d['P'][0]
    sel[ds]=dict(n=len(d['t']),carbon=float((Xr/LITV['YXS']+P/LITV['YPS'])/dS))
json.dump(dict(ident=out,sel=sel),open('ident.json','w'),indent=1); print(sel)
