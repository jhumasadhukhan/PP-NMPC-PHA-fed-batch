import numpy as np, json
from calib2 import *
from ident import fim_ci
from refexp import sim_ref
SEL=['DS01','DS08','DS11','DS12']; CAL=['DS01','DS11','DS12']; VAL=['DS08']; FREE=['mu_xs','mu_ps']
out={'free':FREE,'fixed':{n:LITV[n] for n in NAMES if n not in FREE},'selected':SEL,'cal':CAL,'val':VAL}
def pack(p,runs_cal,runs_all,base=TH_LIT):
    ci,corr,n,cond=fim_ci(p,FREE,runs_cal,base); t=r2_table(p,FREE,runs_all,base)
    return dict(est=dict(zip(FREE,map(float,p))),ci95=dict(zip(FREE,map(float,ci))),corr=float(corr[0,1]),n=int(n),r2={k[0]+'_'+k[1]:v for k,v in t.items()},theta=full_theta(p,FREE,base)[:,0].tolist())
# screening: literature values without fitting and single-run reduced fits for all C. necator runs
ALLC=['DS01','DS02','DS03','DS04','DS07','DS08','DS09','DS11','DS12']
out['screen']={}
for r in ALLC:
    p,J=fit([r],FREE,maxiter=150); out['screen'][r]=dict(J=J,**pack(p,[r],[r])); print('screen',r,{k:round(v,3) for k,v in out['screen'][r]['r2'].items()},flush=True)
# primary calibration / validation
p,J=fit(CAL,FREE,maxiter=300); out['primary']=dict(J=J,**pack(p,CAL,SEL)); print('primary',out['primary']['est'],out['primary']['ci95'],{k:round(v,3) for k,v in out['primary']['r2'].items()},flush=True)
# leave-one-run-out
out['loo']={}
for h in SEL:
    cal=[r for r in SEL if r!=h]; p,J=fit(cal,FREE,maxiter=300); out['loo'][h]=dict(J=J,cal=cal,**pack(p,cal,SEL)); print('loo',h,out['loo'][h]['est'],{k:round(v,3) for k,v in out['loo'][h]['r2'].items() if k.startswith(h)},flush=True)
# published parameter values of Mozumder et al., nothing estimated, on the four held-out runs
t=r2_table([],[],SEL,TH_LIT); out['published']={k[0]+'_'+k[1]:v for k,v in t.items()}; print('published',{k:round(v,3) for k,v in out['published'].items()},flush=True)
# all four selected runs together (final literature estimate)
p,J=fit(SEL,FREE,maxiter=300); out['all4']=dict(J=J,**pack(p,SEL,SEL)); print('all4',out['all4']['est'],flush=True)
# reference experiment
p,J=fit(['REF'],FREE,maxiter=300); out['ref']=dict(J=J,**pack(p,['REF'],['REF'])); print('ref',out['ref']['est'],out['ref']['ci95'],out['ref']['r2'],flush=True)
t=r2_table([],[],['REF'],TH_NOM); out['ref_original']={k[0]+'_'+k[1]:v for k,v in t.items()}; print('ref original 20-par',out['ref_original'])
t=r2_table(out['primary']['est'].values() and list(out['primary']['est'].values()),FREE,['REF']); out['ref_with_lit']={k[0]+'_'+k[1]:v for k,v in t.items()}
# extrapolation checks on runs not selected
p=list(out['primary']['est'].values()); t=r2_table(p,FREE,['DS02','DS07','DS09']); out['not_selected']={k[0]+'_'+k[1]:v for k,v in t.items()}; print('not selected',{k:round(v,3) for k,v in out['not_selected'].items()})
json.dump(out,open('calib_reduced.json','w'),indent=1)
