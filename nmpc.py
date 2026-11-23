import numpy as np, casadi as ca
from model import *
XD=np.array([9.5,2.0,5.0,0.01,3.0])
Q=np.array([59,0.2,5,0.2,0.2]); R=np.array([2.0,2.0])
RINF=np.array([0.1,2.3375,0.3,0.15,3.0]); L=np.array([0.01,0.0,0.05,0.03,0.0])
X0=np.array([0.0,0.13,12.0,0.64,0.6])
def rho(t,r0): return (r0-RINF)*np.exp(-L*t)+RINF
class Ctrl:
    def __init__(s,pp=True,Np=4,dt=1.0,nsub=4,umax=(0.5,0.01),th=TH_NOM,uscale=(1.0,1.0),margin=1e-3):
        s.pp=pp; s.Np=Np; s.dt=dt
        opti=ca.Opti(); X=opti.variable(5,Np+1); U=opti.variable(2,Np)
        x0=opti.parameter(5); t0=opti.parameter(); r0=opti.parameter(5)
        thp=opti.parameter(20)
        J=0; opti.subject_to(X[:,0]==x0)
        def f(x,u): return ca.vertcat(*rhs(x,u,thp,ca))
        us=ca.DM(uscale)
        for k in range(Np):
            xk=X[:,k]; uk=U[:,k]*us; h=dt/nsub
            for j in range(nsub):
                k1=f(xk,uk); k2=f(xk+h/2*k1,uk); k3=f(xk+h/2*k2,uk); k4=f(xk+h*k3,uk)
                xk=xk+h/6*(k1+2*k2+2*k3+k4)
            opti.subject_to(X[:,k+1]==xk)
            tk=t0+(k+1)*dt
            rk=(r0-RINF)*ca.exp(-ca.DM(L)*(k+1)*dt)+RINF
            e=X[:,k+1]-XD
            if pp:
                xi=e/rk
                opti.subject_to(opti.bounded(-1+margin,xi,1-margin))
                eps=ca.log((1+xi)/(1-xi))
                J+=dt*(ca.sum1(ca.DM(Q)*eps**2)+ca.sum1(ca.DM(R)*U[:,k]**2))
            else:
                J+=dt*(ca.sum1(ca.DM(Q)*e**2)+ca.sum1(ca.DM(R)*U[:,k]**2))
            opti.subject_to(X[:,k+1]>=0); opti.subject_to(X[4,k+1]<=6.0)
            opti.subject_to(opti.bounded(0,uk[0],umax[0])); opti.subject_to(opti.bounded(0,uk[1],umax[1]))
        opti.minimize(J)
        opti.solver('ipopt',{'print_time':0,'ipopt.print_level':0,'ipopt.max_iter':300,'ipopt.tol':1e-7,'ipopt.sb':'yes'})
        s.o=opti; s.X=X; s.U=U; s.x0=x0; s.t0=t0; s.r0=r0; s.thp=thp; s.us=np.array(uscale); s.th=th; s.last=None
    def solve(s,x,t,r0):
        o=s.o; o.set_value(s.x0,x); o.set_value(s.t0,t); o.set_value(s.r0,r0); o.set_value(s.thp,s.th)
        if s.last is None:
            o.set_initial(s.X,np.tile(np.array(x)[:,None],(1,s.Np+1))); o.set_initial(s.U,0.0)
        else:
            Xl,Ul=s.last; o.set_initial(s.X,np.hstack([Xl[:,1:],Xl[:,-1:]])); o.set_initial(s.U,np.hstack([Ul[:,1:],Ul[:,-1:]]))
        try:
            sol=o.solve(); Xv=sol.value(s.X); Uv=sol.value(s.U).reshape(2,-1); ok=True
        except RuntimeError:
            ok=False
            if s.last is not None:   # fall back on the shifted previous plan
                Xl,Ul=s.last; Xv=np.hstack([Xl[:,1:],Xl[:,-1:]]); Uv=np.hstack([Ul[:,1:],Ul[:,-1:]])
            else:
                Xv=np.tile(np.array(x)[:,None],(1,s.Np+1)); Uv=np.zeros((2,s.Np))
        s.last=(Xv,Uv); return Uv[:,0]*s.us, ok
def run(pp=True,x0=X0,th_plant=TH_NOM,tf=48,est=None,dist=None,k0=1.25,ctrl=None,**kw):
    c=ctrl or Ctrl(pp=pp,**kw); c.last=None
    e0=np.abs(x0-XD); r0=np.maximum(k0*e0,RINF)
    r0=np.where(L>0,r0,RINF)
    x=np.array(x0,float); xs=[x.copy()]; us=[]; nfail=0; viol=0; xh=[]
    for k in range(tf):
        xm=est(x,k,us) if est else x
        xh.append(np.array(xm))
        rk=rho(k,r0)
        # funnel re-initialisation: if the measured state is outside the funnel, widen it to k0*|e|
        rcur=np.maximum(rk,k0*np.abs(xm-XD)) if pp else rk
        u,ok=c.solve(np.maximum(xm,0),float(k),rcur); nfail+=(not ok)
        n=20; h=1.0/n
        for j in range(n):
            x=rk4_np(x,u,th_plant,h,1)
            if dist is not None: x=x+h*dist(x,k+j*h)
            x=np.maximum(x,0)
        xs.append(x.copy()); us.append(u)
        rk=rho(k+1,r0)
        if np.any(np.abs(x-XD)>=rk): viol+=1
    xs=np.array(xs); us=np.array(us)
    return dict(x=xs,u=us,w=xs[-1,0]/(xs[-1,0]+xs[-1,1])*100,nfail=nfail,viol=viol,r0=r0,xh=np.array(xh))
if __name__=='__main__':
    for pp in (True,False):
        r=run(pp=pp); x=r['x']
        print('PP' if pp else 'NMPC','w=%.2f'%r['w'],'fail',r['nfail'],'viol',r['viol'],'final',np.round(x[-1],3),'CXmax %.2f'%x[:,1].max(),'S30 %.2f N24 %.2f'%(x[30,2],x[24,3]),'FSmax %.3f FNmax(uL/h) %.1f'%(r['u'][:,0].max(),r['u'][:,1].max()*1e6))
