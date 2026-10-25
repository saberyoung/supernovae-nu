from common import *
# Idealized versus joint-IceCube delta-z thresholds at 50 and 80% power.
tab=pd.read_csv(RESULTS/'power_thresholds.csv')
d=tab[tab.target_power_percent.isin([50,80])].copy()
scenarios=['Compact 10-NN region','Broad PC1','Ic-BL subtype']
fig,ax=plt.subplots(figsize=(6.6,4.3)); x=np.arange(len(scenarios)); width=.18
for j,powv in enumerate([50,80]):
    q=d[d.target_power_percent==powv].set_index('label').loc[scenarios]
    ax.bar(x+(j*2-1.5)*width,q.delta_z_synthetic_independent,width,label=f'{powv}% idealized')
    ax.bar(x+(j*2-.5)*width,q.delta_z_joint_icecube,width,label=f'{powv}% joint')
ax.set_xticks(x,scenarios,rotation=12,ha='right')
ax.set_ylabel(r'$\Delta z$ required')
ax.set_title('Realistic detector background reduces absolute sensitivity')
ax.legend(frameon=False,fontsize=7,ncol=2)
fig.tight_layout(); save(fig,'FigureS2_power_realism')
