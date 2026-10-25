from common import *
a=pd.read_csv(RESULTS/'robustness.csv')
b=pd.read_csv(RESULTS/'robustness_independent.csv')
labelmap={
'primary_PC12_k10':'Primary: PC1--PC2, k=10',
'PC12_k5':'Neighborhood k=5','PC12_k15':'Neighborhood k=15','PC12_k20':'Neighborhood k=20',
'PC123_k10':'PC1--PC2--PC3',
'mpeak_le_18p5_PC12_k10':r'$m_{\rm peak}\leq18.5$',
'r_peak_only_PC12_k10':r'$r$-band peaks only',
'endpoint_buffer100_PC12_k10':'100 d endpoint buffer',
'exclude_SLSNI_PC12_k10':'Exclude SLSN-I'}
labels=[labelmap.get(x,x) for x in a.branch]+b.branch.tolist()
p=a.global_p.tolist()+b.global_p.tolist()
group=['publication 2000/10000']*len(a)+['independent 200/500']*len(b)
# Add visual gaps between primary, geometry, selection, independent branches.
y=[]; cur=0
for i in range(len(labels)):
    if i in (1,5,9): cur += .55
    y.append(cur); cur += 1
y=np.array(y)[::-1]
fig,ax=plt.subplots(figsize=(7.4,5.2))
for g,marker in [('publication 2000/10000','o'),('independent 200/500','s')]:
    idx=[i for i,x in enumerate(group) if x==g]
    ax.scatter([p[i] for i in idx],[y[i] for i in idx],marker=marker,s=45,label=g)
ax.axvline(.05,ls='--',lw=1.0,alpha=.8)
ax.axvline(.01,ls=':',lw=.9,alpha=.7)
ax.set_yticks(y,labels); ax.set_xlabel('Global p-value'); ax.set_xlim(0,1)
ax.legend(frameon=False,fontsize=8,loc='upper right')
ax.set_title('Robustness of the localized null result')
fig.tight_layout(); save(fig,'Figure4_robustness')
