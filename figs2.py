import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from calib2 import *
from refexp import sim_ref, TX,XT,TP,PP
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.grid':True,'grid.alpha':0.3,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'savefig.bbox':'tight'})
CB,CR,CG,CO='#1f4e79','#c0392b','#1e7b3c','#d68910'
cr=json.load(open('calib_reduced.json')); R=json.load(open('results2_main.json')); LAM=[0.1,0.25,0.5,1.0]
PL=['REF-R','LIT-4','DS01','DS08','DS11','DS12']
# Fig S1: four held-out runs; published parameters (dashed) and two rates estimated from the other three runs (solid)
fig,ax=plt.subplots(4,4,figsize=(7.2,6.4))
for r,ds in enumerate(cr['selected']):
    d=run_data(ds); tt=np.linspace(0,d['t'].max(),int(d['t'].max())+1)
    yp=sim_batch(TH_LIT[:,None],d['x0'],tt)[:,:,0]; yl=sim_batch(np.array(cr['loo'][ds]['theta'])[:,None],d['x0'],tt)[:,:,0]
    for c,(nm,fp,fl,meas,col,k) in enumerate((('$X_t$',yp[:,0]+yp[:,1],yl[:,0]+yl[:,1],d['X'],CB,'X'),('$C_P$',yp[:,0],yl[:,0],d['P'],CR,'P'),('$C_S$',yp[:,2],yl[:,2],d['S'],CG,'S'),('$C_N$',yp[:,3],yl[:,3],d['N'],CO,'N'))):
        a=ax[r,c]; a.plot(tt,fp,'--',color=col,lw=1.0,label='published parameters'); a.plot(tt,fl,color=col,lw=1.2,label='two rates from other runs'); a.plot(d['t'],meas,'o',mfc='none',mec=col,ms=3.5,label='data')
        a.set_title('%s, %s: R² = %.2f / %.2f'%(ds,nm,cr['published'][ds+'_'+k],cr['loo'][ds]['r2'][ds+'_'+k]),fontsize=6.3)
        if r==3: a.set_xlabel('Time (h)')
        if c==0: a.set_ylabel('g L$^{-1}$')
        if r==0 and c==0: a.legend(fontsize=4.5,frameon=False)
fig.tight_layout(); fig.savefig('figs2/FigS1.png'); plt.close(fig)
# Fig S2: reference experiment, original and reduced parameter sets
fig,ax=plt.subplots(1,3,figsize=(7.2,2.5))
for th_,lab,ls in ((TH_NOM,'Table 1 (20 estimated)','--'),(np.array(cr['ref']['theta']),'Reduced (2 estimated)','-')):
    ts,o=sim_ref(th_); ax[0].plot(ts,o[:,0]+o[:,1],color=CB,ls=ls,lw=1.2,label=lab); ax[1].plot(ts,o[:,0],color=CR,ls=ls,lw=1.2,label=lab); ax[2].plot(ts[5:],100*o[5:,0]/(o[5:,0]+o[5:,1]),color='0.2',ls=ls,lw=1.2,label=lab)
ax[0].plot(TX,XT,'o',mfc='none',mec=CB,ms=4,label='Data [40]'); ax[1].plot(TP,PP,'s',mfc='none',mec=CR,ms=4,label='Data [40]')
ix=[list(TX).index(t) for t in TP[1:]]; ax[2].plot(TP[1:],100*PP[1:]/XT[ix],'^',mfc='none',mec='0.2',ms=4,label='Data [40]')
for a,yl in zip(ax,('Total biomass $X_t$ (g L$^{-1}$)','PHA $C_P$ (g L$^{-1}$)','PHA content (%)')): a.set_xlabel('Time (h)'); a.set_ylabel(yl); a.legend(frameon=False,fontsize=5.8)
fig.tight_layout(); fig.savefig('figs2/FigS2.png'); plt.close(fig)
# Fig 9
fig,ax=plt.subplots(1,2,figsize=(7.0,2.9)); a=ax[0]
for j,lam in enumerate(LAM):
    pp=[R['mismatch']['%s_%g'%(d,lam)]['PP']['w'] for d in PL]; mp=[R['mismatch']['%s_%g'%(d,lam)]['MPC']['w'] for d in PL]
    a.plot([j-0.12]*6,pp,'o',color=CR,ms=3.5,alpha=0.75,label='PP-NMPC' if j==0 else None); a.plot([j+0.12]*6,mp,'s',color=CB,ms=3.5,alpha=0.75,mfc='none',label='NMPC' if j==0 else None)
    a.plot([j-0.24,j],[np.mean(pp)]*2,color=CR,lw=1.5); a.plot([j,j+0.24],[np.mean(mp)]*2,color=CB,lw=1.5)
a.axhline(75,color='0.4',ls=':',lw=0.8); a.text(-0.3,73.2,'reference experiment',ha='left',fontsize=6,color='0.3')
a.set_xticks(range(4)); a.set_xticklabels(['0.1','0.25','0.5','1']); a.set_xlabel('Mismatch size $\\lambda$'); a.set_ylabel('Final PHA content (%)'); a.legend(frameon=False,fontsize=6.5,loc='lower left'); a.set_title('(a)',loc='left',fontsize=8,fontweight='bold')
a=ax[1]; Lg=np.load('soft2_log_PP.npy'); tt=np.arange(len(Lg))
a.plot(tt,Lg[:,0],color=CR,lw=1.3,label='$C_P$ true'); a.plot(tt,Lg[:,2],'o',color=CR,ms=2.5,mfc='none',label='$C_P$ estimated')
a.plot(tt,Lg[:,1],color=CG,lw=1.3,label='$C_X$ true'); a.plot(tt,Lg[:,3],'s',color=CG,ms=2.5,mfc='none',label='$C_X$ estimated')
a.set_xlabel('Time (h)'); a.set_ylabel('Concentration (g L$^{-1}$)'); a.legend(frameon=False,fontsize=6.5); a.set_title('(b)',loc='left',fontsize=8,fontweight='bold')
fig.tight_layout(); fig.savefig('figs2/Fig9.png'); plt.close(fig)
# Fig S3: nominal closed-loop runs of the implementation
from nmpc import XD,RINF,L,X0,rho
tp=np.load('traj2_nom_PP_main.npy'); tm=np.load('traj2_nom_MPC_main.npy'); t=np.arange(49)
def full(tr): return np.vstack([X0,tr[:,:5]])
xp,xm=full(tp),full(tm); r0=np.where(L>0,1.25*np.abs(X0-XD),RINF); rr=np.array([rho(k,r0) for k in t])
fig,ax=plt.subplots(2,3,figsize=(7.2,4.0)); ax=ax.ravel()
labs=['$C_P$ (g L$^{-1}$)','$C_X$ (g L$^{-1}$)','$C_S$ (g L$^{-1}$)','$C_N$ (g L$^{-1}$)','$V$ (L)']
for i in range(5):
    a=ax[i]; a.fill_between(t,np.maximum(XD[i]-rr[:,i],0),XD[i]+rr[:,i],color='0.9'); a.plot(t,xp[:,i],color=CR,lw=1.3,label='PP-NMPC'); a.plot(t,xm[:,i],color=CB,lw=1.3,ls='--',label='NMPC'); a.set_ylabel(labs[i]); a.set_xlabel('Time (h)')
a=ax[5]; a.plot(t,100*xp[:,0]/(xp[:,0]+xp[:,1]),color=CR,lw=1.3,label='PP-NMPC'); a.plot(t,100*xm[:,0]/(xm[:,0]+xm[:,1]),color=CB,lw=1.3,ls='--',label='NMPC'); a.set_ylabel('PHA content (%)'); a.set_xlabel('Time (h)'); a.legend(frameon=False,fontsize=6.5)
fig.tight_layout(); fig.savefig('figs2/FigS3.png'); plt.close(fig)
# Fig S4: soft-sensor parity on the test trajectories
P=np.load('ann_test_pred.npy'); am=json.load(open('ann_metrics.json'))
fig,ax=plt.subplots(1,2,figsize=(5.6,2.7))
for i,(nm,col,k) in enumerate((('$C_X$',CG,'CX'),('$C_P$',CR,'CP'))):
    a=ax[i]; a.plot(P[:,i],P[:,i+2],'.',ms=1.5,color=col,alpha=0.4); m=max(P[:,i].max(),P[:,i+2].max()); a.plot([0,m],[0,m],'k-',lw=0.7)
    a.set_xlabel('Model %s (g L$^{-1}$)'%nm); a.set_ylabel('ANN estimate (g L$^{-1}$)'); a.set_title('%s: R² = %.3f, RMSE = %.3f g L$^{-1}$'%(nm,am['7in']['R2_'+k],am['7in']['RMSE_'+k]),fontsize=7)
fig.tight_layout(); fig.savefig('figs2/FigS4.png'); plt.close(fig)
print('ok')
