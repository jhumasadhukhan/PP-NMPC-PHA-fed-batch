import numpy as np, pickle, json
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from model import *
rng=np.random.default_rng(0)
def rand_ic():
    CS=rng.uniform(5,20); CN=CS/rng.uniform(20,40); CX=rng.uniform(0.05,1.0)
    return np.array([0.0,CX,CS,CN,0.6])
def smooth_profile(n,hi):
    # random smooth non-negative feed profile, piecewise constant per hour
    k=rng.integers(3,7); knots=rng.uniform(0,hi,k)*rng.choice([0,1,1,1],k); xs=np.linspace(0,n-1,k)
    return np.clip(np.interp(np.arange(n),xs,knots)*rng.uniform(0.5,1.2,n),0,None)
def simulate(n_traj=400,tf=48,th=TH_NOM):
    rows=[]
    for j in range(n_traj):
        x=rand_ic(); S0,N0=x[2],x[3]
        FS=smooth_profile(tf,0.13); FN=smooth_profile(tf,100e-6)*rng.choice([0,1],p=[0.3,0.7])
        cS=cN=0.0
        rows.append([j,0,S0,N0,x[2],x[3],0,0,x[1],x[0]])
        for k in range(tf):
            u=[FS[k],FN[k]]
            for _ in range(20): x=np.maximum(rk4_np(x,u,th,0.05,1),0)
            if x[4]>6: break
            cS+=FS[k]*CSF; cN+=FN[k]*CNF
            rows.append([j,k+1,S0,N0,x[2],x[3],cS/x[4],cN/x[4],x[1],x[0]])
    return np.array(rows)
if __name__=='__main__':
    D=simulate(); np.save('ann_data.npy',D)
    traj=D[:,0]; ids=np.unique(traj); rng2=np.random.default_rng(1); rng2.shuffle(ids); test=set(ids[:int(0.2*len(ids))])
    te=np.array([t in test for t in traj]); tr=~te
    out={}
    for name,cols in (('7in',[1,2,3,4,5,6,7]),('5in',[1,2,3,4,5])):
        Xs=MinMaxScaler().fit(D[tr][:,cols]); Ys=MinMaxScaler().fit(D[tr][:,8:10])
        m=MLPRegressor(hidden_layer_sizes=(128,64,32),activation='relu',alpha=1e-3,learning_rate_init=1e-3,max_iter=3000,early_stopping=True,n_iter_no_change=60,random_state=0)
        m.fit(Xs.transform(D[tr][:,cols]),Ys.transform(D[tr][:,8:10]))
        P=Ys.inverse_transform(m.predict(Xs.transform(D[te][:,cols]))); Y=D[te][:,8:10]
        r2=1-((Y-P)**2).sum(0)/((Y-Y.mean(0))**2).sum(0); rmse=np.sqrt(((Y-P)**2).mean(0)); mae=np.abs(Y-P).mean(0)
        out[name]=dict(R2_CX=r2[0],R2_CP=r2[1],RMSE_CX=rmse[0],RMSE_CP=rmse[1],MAE_CX=mae[0],MAE_CP=mae[1],n_train=int(tr.sum()),n_test=int(te.sum()),n_traj=len(ids))
        print(name,{k:(round(v,4) if isinstance(v,float) else v) for k,v in out[name].items()},flush=True)
        if name=='7in': pickle.dump((m,Xs,Ys),open('ann7.pkl','wb')); np.save('ann_test_pred.npy',np.hstack([Y,P]))
    json.dump(out,open('ann_metrics.json','w'))
