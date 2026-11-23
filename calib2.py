import numpy as np, pandas as pd, json
from scipy.optimize import differential_evolution
from model import *
from refexp import TX,XT,TP,PP
NFRAC=28.014/132.14            # g N per g (NH4)2SO4
# literature values of Mozumder et al. (same model structure, C. necator DSM 545, glucose)
LITV=dict(mu_xs=0.46,mu_xp=0.126,mu_ps=0.217,KS=1.2,KIS=17.43,KN=0.254,KIN=1.5,KP=0.148,KPS=4.1,KPIS=80.0,KPIN=0.262,fmax=3.3,CXmax=68.0,alpha=5.8,beta=3.85,YXS=0.48,YPS=0.30,YXN=8.9,YXP=0.88,ms=0.02)
TH_LIT=np.array([LITV[n] for n in NAMES])
df=pd.read_csv('master.csv'); lit=df[~df.id.str.startswith('SYN')].copy(); lit['N']=lit['N']*NFRAC
def run_data(ds):
    d=lit[lit.id==ds]; return dict(t=d.t.values,X=d.X.values,P=d.P.values,S=d.S.values,N=d.N.values,x0=[d.P.values[0],d.X.values[0]-d.P.values[0],d.S.values[0],d.N.values[0],1.0])
def sim_batch(TH,x0,tgrid,dt=0.1):
    S=TH.shape[1]; x=np.tile(np.array(x0,float)[:,None],(1,S)); out=[]; t=0.0; u=[0.0,0.0]
    for tg in tgrid:
        for _ in range(int(round((tg-t)/dt))):
            k1=np.array(rhs(x,u,TH)); k2=np.array(rhs(x+dt/2*k1,u,TH)); k3=np.array(rhs(x+dt/2*k2,u,TH)); k4=np.array(rhs(x+dt*k3,u,TH))
            x=np.maximum(x+dt/6*(k1+2*k2+2*k3+k4),0); x[1]=np.maximum(x[1],1e-9)
        t=tg; out.append(x.copy())
    return np.array(out)
def sim_ref_vec(TH,tgrid,dt=0.05,S_set=5.0,N_set=0.64,x0=(0.0,0.13,12.0,0.64,0.6)):
    """reference fed-batch [40]: substrate held at 5 g/L once reached, nitrogen held at 0.64 g/L (vectorised over parameter sets)"""
    S=TH.shape[1]; x=np.tile(np.array(x0,float)[:,None],(1,S)); out=[]; t=0.0
    for tg in tgrid:
        for _ in range(int(round((tg-t)/dt))):
            mxs,mxp,mps=rates(x,TH); CP,CX,CS,CN,V=x
            qS=(mxs/TH[15]+mps/TH[16]+TH[19])*CX; qN=(mxs+mxp)/TH[17]*CX
            holdS=CS<=S_set; holdN=CN<=N_set
            a11=np.where(holdS,CSF-BS*CS,1.0); a12=np.where(holdS,-BN*CS,0.0); b1=np.where(holdS,qS*V,0.0)
            a21=np.where(holdN,-BS*CN,0.0); a22=np.where(holdN,CNF-BN*CN,1.0); b2=np.where(holdN,qN*V,0.0)
            det=a11*a22-a12*a21; FS=np.maximum((b1*a22-a12*b2)/det,0); FN=np.maximum((a11*b2-a21*b1)/det,0)
            Fv=FS*BS+FN*BN; D=Fv/V
            dx=np.array([(mps-mxp/TH[18])*CX-D*CP,(mxs+mxp-D)*CX,FS*CSF/V-D*CS-qS,FN*CNF/V-D*CN-qN,Fv])
            x=np.maximum(x+dt*dx,0); x[1]=np.maximum(x[1],1e-9)
        t=tg; out.append(x.copy())
    return np.array(out)
TREF=np.unique(np.concatenate([TX,TP])); IX=np.searchsorted(TREF,TX); IP=np.searchsorted(TREF,TP)
def nse(y,m): return float(1-np.sum((y-m)**2)/np.sum((y-y.mean())**2))
def full_theta(pfree,free,base=TH_LIT):
    pf=np.atleast_2d(np.asarray(pfree).T).T if np.ndim(pfree)==1 else pfree
    TH=np.tile(base[:,None],(1,pf.shape[1])).astype(float)
    for i,n in enumerate(free): TH[NAMES.index(n)]=pf[i]
    return TH
def residual_terms(TH,runs):
    """list of (run, variable, measured, modelled[n,S])"""
    out=[]
    for r in runs:
        if r=='REF':
            y=sim_ref_vec(TH,TREF); out+= [(r,'X',XT,(y[:,0]+y[:,1])[IX]),(r,'P',PP,y[:,0][IP])]
        else:
            d=run_data(r); y=sim_batch(TH,d['x0'],d['t']); out+= [(r,'X',d['X'],y[:,0]+y[:,1]),(r,'P',d['P'],y[:,0]),(r,'S',d['S'],y[:,2]),(r,'N',d['N'],y[:,3])]
    return out
def objective(runs,free,base=TH_LIT):
    def obj(p):
        TH=full_theta(p,free,base); J=0; terms=residual_terms(TH,runs)
        for r,v,meas,mod in terms: J=J+np.sqrt(np.mean((mod-meas[:,None])**2,axis=0))/(meas.max()-meas.min())
        J=J/len(terms); return np.where(np.isfinite(J),J,1e6)
    return obj
def fit(runs,free,base=TH_LIT,seed=1,maxiter=300,bounds=None):
    bnd=bounds or [(LB[NAMES.index(n)],UB[NAMES.index(n)]) for n in free]
    r=differential_evolution(objective(runs,free,base),bnd,popsize=15,mutation=(0.5,1.5),recombination=0.7,maxiter=maxiter,tol=1e-6,polish=False,vectorized=True,updating='deferred',seed=seed)
    return r.x,float(r.fun)
def r2_table(p,free,runs,base=TH_LIT):
    TH=full_theta(p,free,base); return {(r,v):nse(meas,mod[:,0]) for r,v,meas,mod in residual_terms(TH,runs)}
if __name__=='__main__':
    C=['DS01','DS02','DS07','DS08','DS09','DS11','DS12']
    FREE=['mu_xs','mu_xp','mu_ps','KS','KN','KP','KPS','KPIN','fmax','ms']
    print('literature values only (no fitting):')
    t=r2_table([],[],['REF']+C); print({k[0]+'_'+k[1]:round(v,3) for k,v in t.items()})
    print('per-run fits, reduced set (10 free), fixed at literature:')
    for r in ['REF']+C:
        p,J=fit([r],FREE,maxiter=200); t=r2_table(p,FREE,[r]); print(r,'J=%.3f'%J,{k[1]:round(v,3) for k,v in t.items()},np.round(p,3),flush=True)
