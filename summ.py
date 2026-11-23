import json, numpy as np
LAM=[0.1,0.25,0.5,1.0]; CN=['DS01','DS02','DS03','DS04','DS07','DS08','DS09','DS11','DS12']
def summ(r,key,CN=None):
    CN=CN or sorted({k.rsplit('_',1)[0] for k in r[key]},key=lambda d:(d.startswith('DS'),d))
    out=[]
    for lam in LAM:
        g=lambda c,f:np.array([r[key]['%s_%g'%(d,lam)][c][f] for d in CN])
        pp,mp=g('PP','w'),g('MPC','w')
        out.append(dict(lam=lam,pp_mean=pp.mean(),pp_min=pp.min(),pp_max=pp.max(),mp_mean=mp.mean(),mp_min=mp.min(),mp_max=mp.max(),dmean=(pp-mp).mean(),dmin=(pp-mp).min(),dmax=(pp-mp).max(),wins=int((pp>mp).sum()),above75_pp=int((pp>75).sum()),above75_mp=int((mp>75).sum()),vpp=g('PP','viol').mean(),vmp=g('MPC','viol').mean(),fpp=int(g('PP','nfail').sum()),fmp=int(g('MPC','nfail').sum()),
            **({'rCP_pp':g('PP','rmse_CP').mean(),'rCX_pp':g('PP','rmse_CX').mean()} if key=='soft_mis' else {})))
    return out
if __name__=='__main__':
    for tag in ('main','N1e-3','N1e-2'):
        r=json.load(open('results2_%s.json'%tag)); print(r.get('renominal')); print('=====',tag,{k:round(v['w'],2) for k,v in r['nominal'].items()},{k:v['viol'] for k,v in r['nominal'].items()})
        for key in ('mismatch','soft_mis'):
            if key not in r: continue
            print(' ',key)
            for s in summ(r,key): print('   ',{k:(round(float(v),2) if not isinstance(v,int) else v) for k,v in s.items()})
        if 'soft_nom' in r:
            for n in ('PP','MPC'):
                a=r['soft_nom'][n]; w=np.array([x['w'] for x in a]); print('  soft nominal',n,'mean %.2f sd %.2f min %.2f max %.2f'%(w.mean(),w.std(ddof=1),w.min(),w.max()),'viol %.2f'%np.mean([x['viol'] for x in a]),'fail',sum(x['nfail'] for x in a),'rmseCP %.3f rmseCX %.3f'%(np.mean([x['rmse_CP'] for x in a]),np.mean([x['rmse_CX'] for x in a])))
