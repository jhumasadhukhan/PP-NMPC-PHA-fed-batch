import numpy as np
from model import *
# data digitised from Fig. 2 of the manuscript (reference fed-batch experiment [40])
TX=np.array([0,12,15,21,24,36,39,42,45.]); XT=np.array([0.13,1.40,2.15,4.28,5.92,8.40,8.83,9.95,9.36])
TP=np.array([0,15,24,36,45.]); PP=np.array([0.0,0.72,3.35,5.78,7.12])
def sim_ref(th,tf=45,dt=0.02,S_set=5.0,N_set=0.64,x0=(0.0,0.13,12.0,0.64,0.6),csf=CSF,ret_feed=False):
    """fed-batch with substrate held at S_set once it has fallen to it and nitrogen held at N_set (ideal feedback feeding)"""
    x=np.array(x0,float); out=[x.copy()]; ts=[0.0]; n=int(round(tf/dt)); fs=fn=0.0; feeds=[]
    for k in range(n):
        mxs,mxp,mps=rates(x,th); CP,CX,CS,CN,V=x
        qS=(mxs/th[15]+mps/th[16]+th[19])*CX; qN=(mxs+mxp)/th[17]*CX
        # solve for FS, FN holding dCS/dt=0 (if at setpoint) and dCN/dt=0:  FS*csf/V - D*CS - qS = 0 ; FN*CNF/V - D*CN - qN = 0 ; D=(FS*BS'+FN*BN)/V
        bs=(RFS-csf)/RW
        A=np.array([[csf-bs*CS, -BN*CS],[-bs*CN, CNF-BN*CN]]); b=np.array([qS*V if CS<=S_set else 0.0, qN*V if CN<=N_set else 0.0])
        if CS>S_set: A[0]=[1,0]
        if CN>N_set: A[1]=[0,1]
        F=np.maximum(np.linalg.solve(A,b),0); FS,FN=F
        Fv=FS*bs+FN*BN; D=Fv/V
        dx=np.array([(mps-mxp/th[18])*CX-D*CP,(mxs+mxp-D)*CX,FS*csf/V-D*CS-qS,FN*CNF/V-D*CN-qN,Fv])
        x=np.maximum(x+dt*dx,0); out.append(x.copy()); ts.append((k+1)*dt); feeds.append((FS,FN))
    out=np.array(out); ts=np.array(ts)
    return (ts,out,np.array(feeds)) if ret_feed else (ts,out)
def r2(y,m): return 1-np.sum((y-m)**2)/np.sum((y-y.mean())**2)
def ref_metrics(th,**kw):
    ts,o=sim_ref(th,**kw); xt=np.interp(TX,ts,o[:,0]+o[:,1]); p=np.interp(TP,ts,o[:,0])
    return r2(XT,xt),r2(PP,p),xt,p,o[-1]
if __name__=='__main__':
    for csf in (62.93,650.0):
        a,b,xt,p,xf=ref_metrics(TH_NOM,csf=csf); print('C_SF',csf,'R2 X %.4f P %.4f'%(a,b),'Xt',np.round(xt,2),'P',np.round(p,2),'final',np.round(xf,2),'w %.1f'%(100*xf[0]/(xf[0]+xf[1])))
