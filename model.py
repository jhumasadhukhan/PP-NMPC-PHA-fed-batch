import numpy as np, casadi as ca
NAMES=['mu_xs','mu_xp','mu_ps','KS','KIS','KN','KIN','KP','KPS','KPIS','KPIN','fmax','CXmax','alpha','beta','YXS','YPS','YXN','YXP','ms']
TH_NOM=np.array([0.14884,0.49958,2.28684,8.71648,16.54088,1.31361,36.1489,0.12512,19.89946,46.6806,1.29648,9.79296,30.5196,0.5,7.97204,1.04905,0.17857,13.34029,2.8021,0.02614])
LB=np.array([0.01,0.005,0.05,0.1,5,0.01,1,0.01,0.5,10,0.01,1,10,0.5,0.5,0.1,0.05,2,0.1,0.01])
UB=np.array([1.5,1.0,5,10,100,2,50,1,20,300,2,15,200,20,15,1.5,1.5,30,3,0.2])
CSF,CNF,RFS,RFN,RW=62.93,164.0,1230.0,1040.0,1000.0
BS=(RFS-CSF)/RW; BN=(RFN-CNF)/RW
def rates(x,th,np_=np):
    """x=[CP,CX,CS,CN,V]; works for numpy arrays (th may be array of shape (20,) or (20,n)) and casadi"""
    CP,CX,CS,CN=x[0],x[1],x[2],x[3]
    (mu_xs,mu_xp,mu_ps,KS,KIS,KN,KIN,KP,KPS,KPIS,KPIN,fmax,CXmax,al,be,YXS,YPS,YXN,YXP,ms)=[th[i] for i in range(20)]
    if np_ is np:
        CXp=np.maximum(CX,1e-9); CSp=np.maximum(CS,0); CNp=np.maximum(CN,0); CPp=np.maximum(CP,0)
        f=CPp/CXp
        lg=1-np.minimum(CXp/CXmax,1.0)**al
        sat=1-np.minimum(f/fmax,1.0)**be
    else:
        CXp=ca.fmax(CX,1e-9); CSp=ca.fmax(CS,0); CNp=ca.fmax(CN,0); CPp=ca.fmax(CP,0)
        f=CPp/CXp
        lg=1-ca.fmin(CXp/CXmax,1.0)**al
        sat=1-ca.fmin(f/fmax,1.0)**be
    hN=CNp/(KN+CNp+CNp**2/KIN)
    mxs=mu_xs*(CSp/(KS+CSp+CSp**2/KIS))*hN*lg
    mxp=mu_xp*(f/(KP+f))*hN*lg
    mps=mu_ps*(KPIN/(CNp+KPIN))*(CSp/(KPS+CSp+CSp**2/KPIS))*sat
    return mxs,mxp,mps
def rhs(x,u,th,np_=np):
    CP,CX,CS,CN,V=x[0],x[1],x[2],x[3],x[4]
    FS,FN=u[0],u[1]
    mxs,mxp,mps=rates(x,th,np_)
    YXS,YPS,YXN,YXP,ms=th[15],th[16],th[17],th[18],th[19]
    F=FS*BS+FN*BN+0*V; D=F/V
    dCP=(mps-mxp/YXP)*CX-D*CP
    dCX=(mxs+mxp-D)*CX
    dCS=FS*CSF/V-D*CS-(mxs/YXS+mps/YPS+ms)*CX
    dCN=FN*CNF/V-D*CN-(mxs+mxp)/YXN*CX
    return [dCP,dCX,dCS,dCN,F]
def rk4_np(x,u,th,dt,n):
    x=np.array(x,float); h=dt/n
    for _ in range(n):
        k1=np.array(rhs(x,u,th)); k2=np.array(rhs(x+h/2*k1,u,th)); k3=np.array(rhs(x+h/2*k2,u,th)); k4=np.array(rhs(x+h*k3,u,th))
        x=x+h/6*(k1+2*k2+2*k3+k4)
    return x
