from common import *
emp = pd.read_csv(RESULTS/'empirical_injection_power.csv')
syn = pd.read_csv(ROOT/'data/idealized_power_benchmark.csv')
scenarios=[('compact_10NN_region','Compact 10-NN region','knn10'),('broad_PC1_upper_tail','Broad PC1','linear_PC1_PC2'),('exact_IcBL_subtype','Ic-BL subtype','max_subtype')]
tests=[('knn10','localized'),('linear_PC1_PC2','gradient'),('max_subtype','subtype')]
fig,axs=plt.subplots(1,3,figsize=(11.5,3.55),sharey=True)
for ax,(sc,title,matched) in zip(axs,scenarios):
    q=emp[(emp.scenario==sc)&(emp.alpha==.01)]
    for test,label in tests:
        z=q[q.test==test].sort_values('delta_z')
        ax.plot(z.delta_z,z.power,marker='o',ms=3,label=label)
    s=syn[(syn.scenario==sc)&(syn.test==matched)].sort_values('mean_shift_delta')
    ax.plot(s.mean_shift_delta,s.power_at_global_alpha_0p01,ls='--',lw=1.5,label='matched idealized')
    ax.axhline(.5,lw=.7,ls=':',alpha=.6)
    ax.axhline(.8,lw=.8,ls=':',alpha=.8)
    ax.set_title(title); ax.set_xlabel(r'Injected $\Delta z$'); ax.set_ylim(0,1.03)
axs[0].set_ylabel(r'Detection power (global $\alpha=0.01$)')
axs[-1].legend(frameon=False,fontsize=7)
fig.tight_layout(); save(fig,'Figure2_population_power')
