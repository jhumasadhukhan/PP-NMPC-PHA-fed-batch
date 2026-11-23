import numpy as np, json, pickle, sys, warnings
warnings.filterwarnings('ignore')
from nmpc import *
cr=json.load(open('calib_reduced.json'))
PLANTS={'REF-R':cr['ref']['theta'],'LIT-4':cr['all4']['theta'],**{d:cr['screen'][d]['theta'] for d in cr['selected']}}
CNEC=list(PLANTS)
LAM=[0.1,0.25,0.5,1.0]
m,Xs,Ys=pickle.load(open('ann7.pkl','rb'))
class Est:
    def __init__(s,x0,seed,sS=0.1,sN=0.005):
        s.S0,s.N0,s.V0=x0[2],x0[3],x0[4]; s.rng=np.random.default_rng(seed); s.sS,s.sN=sS,sN; s.log=[]
    def __call__(s,x,k,us):
        us=np.array(us).reshape(-1,2)
        cS=us[:,0].sum()*CSF; cN=us[:,1].sum()*CNF; V=s.V0+(us[:,0].sum()*BS+us[:,1].sum()*BN)
        Sm=max(x[2]+s.rng.normal(0,s.sS),0); Nm=max(x[3]+s.rng.normal(0,s.sN),0)
        y=Ys.inverse_transform(m.predict(Xs.transform([[k,s.S0,s.N0,Sm,Nm,cS/V,cN/V]])))[0]
        xh=np.array([max(y[1],0),max(y[0],1e-3),Sm,Nm,V]); s.log.append((x.copy(),xh.copy())); return xh
UMN=float(sys.argv[1]); TAG=sys.argv[2]; MODE=sys.argv[3]
cP=Ctrl(pp=True,nsub=2,umax=(0.5,UMN)); cM=Ctrl(pp=False,nsub=2,umax=(0.5,UMN))
def both(**kw):
    out={}
    for name,c,pp in (('PP',cP,True),('MPC',cM,False)):
        r=run(pp=pp,ctrl=c,**kw); out[name]=r
    return out
res={}
# E1 nominal
b=both(); res['nominal']={k:dict(w=v['w'],viol=v['viol'],nfail=v['nfail'],xf=v['x'][-1].tolist()) for k,v in b.items()}
np.save('traj2_nom_PP_%s.npy'%TAG,np.hstack([b['PP']['x'][1:],b['PP']['u']])); np.save('traj2_nom_MPC_%s.npy'%TAG,np.hstack([b['MPC']['x'][1:],b['MPC']['u']]))
print('nominal',res['nominal'],flush=True)
thre=np.array(PLANTS['REF-R']); res['renominal']={}
for name,pp in (('PP',True),('MPC',False)):
    c=Ctrl(pp=pp,nsub=2,umax=(0.5,UMN),th=thre); r=run(pp=pp,ctrl=c,th_plant=thre)
    res['renominal'][name]=dict(w=r['w'],viol=r['viol'],nfail=r['nfail'],xf=r['x'][-1].tolist()); np.save('traj2_re_%s_%s.npy'%(name,TAG),np.hstack([r['x'][1:],r['u']]))
print('renominal',res['renominal'],flush=True)
# E2 parametric mismatch, true state
res['mismatch']={}
for ds in CNEC:
    thj=np.array(PLANTS[ds])
    for lam in LAM:
        thp=TH_NOM+lam*(thj-TH_NOM); b=both(th_plant=thp)
        res['mismatch']['%s_%g'%(ds,lam)]={k:dict(w=v['w'],viol=v['viol'],nfail=v['nfail'],CP=v['x'][-1,0],CX=v['x'][-1,1]) for k,v in b.items()}
        print(ds,lam,{k:(round(v['w'],2),v['viol'],v['nfail']) for k,v in b.items()},flush=True)
    json.dump(res,open('results2_%s.json'%TAG,'w'))
if MODE!='full': print('DONE'); sys.exit()
# E3 soft sensor: nominal, 20 noise seeds
res['soft_nom']={'PP':[],'MPC':[]}
for seed in range(20):
    for name,c,pp in (('PP',cP,True),('MPC',cM,False)):
        e=Est(X0,seed); r=run(pp=pp,ctrl=c,est=e)
        L_=np.array([[a[0],a[1],b_[0],b_[1]] for a,b_ in e.log])  # true CP,CX, est CP,CX
        res['soft_nom'][name].append(dict(w=r['w'],viol=r['viol'],nfail=r['nfail'],rmse_CP=float(np.sqrt(np.mean((L_[:,0]-L_[:,2])**2))),rmse_CX=float(np.sqrt(np.mean((L_[:,1]-L_[:,3])**2)))))
        if seed==0: np.save('soft2_log_%s.npy'%name,L_); np.save('soft2_traj_%s.npy'%name,np.hstack([r['x'][1:],r['u']]))
    print('soft seed',seed,[round(res['soft_nom'][n][-1]['w'],2) for n in ('PP','MPC')],flush=True)
json.dump(res,open('results2_%s.json'%TAG,'w'))
# E3b soft sensor under parametric mismatch (one noise seed)
res['soft_mis']={}
for ds in CNEC:
    thj=np.array(PLANTS[ds])
    for lam in LAM:
        thp=TH_NOM+lam*(thj-TH_NOM); o={}
        for name,c,pp in (('PP',cP,True),('MPC',cM,False)):
            e=Est(X0,100); r=run(pp=pp,ctrl=c,est=e,th_plant=thp)
            L_=np.array([[a[0],a[1],b_[0],b_[1]] for a,b_ in e.log])
            o[name]=dict(w=r['w'],viol=r['viol'],nfail=r['nfail'],rmse_CP=float(np.sqrt(np.mean((L_[:,0]-L_[:,2])**2))),rmse_CX=float(np.sqrt(np.mean((L_[:,1]-L_[:,3])**2))))
        res['soft_mis']['%s_%g'%(ds,lam)]=o
        print('softmis',ds,lam,{k:(round(v['w'],2),v['viol'],v['nfail']) for k,v in o.items()},flush=True)
    json.dump(res,open('results2_%s.json'%TAG,'w'))
print('DONE')
