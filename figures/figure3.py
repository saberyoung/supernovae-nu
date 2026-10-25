from common import *
nodes=pd.read_csv(RESULTS/'primary_nodes.csv')
hot=pd.read_csv(RESULTS/'hotspot_members.csv')
null=pd.read_csv(ROOT/'results/primary_null_distribution.csv')
T=null['T_local'].to_numpy(float); tobs=float(json.loads((ROOT/'results/primary_summary.json').read_text())['T_observed'])
fig,axs=plt.subplots(1,2,figsize=(8.5,3.7))
ax=axs[0]
sc=ax.scatter(nodes.PC1,nodes.PC2,c=nodes.z_local_calibrated,s=25,cmap='coolwarm',vmin=-3,vmax=3)
ax.scatter(hot.PC1,hot.PC2,s=68,facecolors='none',edgecolors='0.20',linewidths=.85, label='10 contributing neighbours')
center = nodes.loc[nodes.is_primary_hotspot_center].iloc[0]
ax.scatter([center.PC1],[center.PC2],marker='*',s=150,facecolors='none',edgecolors='black',linewidths=1.2,label='geometric centre (score excluded)')
ax.legend(frameon=False, fontsize=6.8, loc='lower right')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('a  Neutrino evidence in optical space',loc='left',fontweight='bold')
fig.colorbar(sc,ax=ax,label=r'calibrated evidence mark $z$')
ax=axs[1]
ax.hist(T,bins=45,density=True,histtype='stepfilled',alpha=.58)
ax.axvline(tobs,ls='--',lw=1.6)
ax.text(.97,.94,rf'$T_{{\rm local}}^{{\rm obs}}={tobs:.3f}$'+'\n'+r'$p_{\rm local}=0.539$',ha='right',va='top',transform=ax.transAxes)
ax.set_xlabel(r'$T_{\rm local}$'); ax.set_ylabel('Background density')
ax.set_title('b  Joint detector null',loc='left',fontweight='bold')
fig.tight_layout(); save(fig,'Figure3_primary_search')
