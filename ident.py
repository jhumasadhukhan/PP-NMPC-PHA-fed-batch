import numpy as np
from calib2 import *
def fim_ci(p,free,runs,base=TH_LIT,rel=1e-3):
    """95% CI half-widths from the Fisher information matrix with weighted least squares (variance per run and variable)"""
    p=np.array(p,float)
    def resid(q):
        TH=full_theta(q,free,base); out=[]
        for r,v,meas,mod in residual_terms(TH,runs): out.append((mod[:,0],meas))
        return out
    r0=resid(p); n=sum(len(m) for _,m in r0); k=len(p)
    sig=[np.sqrt(max(np.sum((m-y)**2)/max(len(y)-1,1),1e-12)) for m,y in r0]
    Jm=np.zeros((n,k))
    for j in range(k):
        h=rel*max(abs(p[j]),1e-6); qp=p.copy(); qp[j]+=h; qm=p.copy(); qm[j]-=h
        rp=resid(qp); rm=resid(qm)
        Jm[:,j]=np.concatenate([(a[0]-b[0])/(2*h)/s for a,b,s in zip(rp,rm,sig)])
    F=Jm.T@Jm; cov=np.linalg.pinv(F); sd=np.sqrt(np.diag(cov)); corr=cov/np.outer(sd,sd)
    return 1.96*sd, corr, n, np.linalg.cond(F)
if __name__=='__main__':
    import sys
    CAL=['DS01','DS11','DS12']; VAL=['DS08','DS09']
    for free in (['mu_xs','mu_ps','KPIN','fmax','KS','KPS'],['mu_xs','mu_ps','KPIN','KS','KPS'],['mu_xs','mu_ps','KPIN','KPS'],['mu_xs','mu_ps','KPIN','KS'],['mu_xs','mu_ps','KPIN'],['mu_xs','mu_ps','KPIN','KN','KS']):
        p,J=fit(CAL,free,maxiter=200); ci,corr,n,cond=fim_ci(p,free,CAL); t=r2_table(p,free,CAL+VAL)
        print(free,'J=%.4f n=%d cond=%.1e'%(J,n,cond)); print('   est',dict(zip(free,np.round(p,3))),' CI%',dict(zip(free,np.round(100*ci/np.abs(p),0))))
        print('   max |corr| off-diag %.2f'%np.max(np.abs(corr-np.eye(len(p)))))
        for r in CAL+VAL: print('     ',r,'CAL' if r in CAL else 'VAL',{k[1]:round(v,3) for k,v in t.items() if k[0]==r})
        sys.stdout.flush()
