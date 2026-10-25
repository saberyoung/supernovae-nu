from common import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

df = pd.read_csv(ROOT/'data/optical_reference.csv')
summary = json.loads((RESULTS/'optical_summary.json').read_text())
null = pd.read_csv(RESULTS/'optical_purity_null.csv').null_purity.to_numpy()
obs = summary['ordinary4_knn10_same_subtype_purity']
exp = summary['ordinary4_knn10_random_expectation']
pperm = summary['ordinary4_knn10_permutation_p']
sil = summary['ordinary4_silhouette_PC12']

fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.0), gridspec_kw={'width_ratios':[1.08,1.0]}, constrained_layout=True)

ax = axs[0]
for typ, g in df.groupby('type', sort=False):
    ax.scatter(g.PC1, g.PC2, s=16, alpha=.72, label=typ)
ax.set_xlabel('PC1 (61.8% variance)')
ax.set_ylabel('PC2 (21.8% variance)')
ax.set_title('a  Frozen optical light-curve space', loc='left', fontweight='bold')
ax.legend(fontsize=6.6, frameon=False, ncol=2, loc='lower right')

# Small loading inset avoids obscuring the dense central population.
iax = inset_axes(ax, width='31%', height='31%', loc='upper right', borderpad=1.0)
loadings = {
    r'$M_{\rm peak}$': (-0.5708, 0.6298),
    r'$\log t_{1/2,\rm rise}$': (0.5499, 0.7697),
    r'$\log t_{1/2,\rm fade}$': (0.6098, -0.1045),
}
for label, (x, y) in loadings.items():
    iax.annotate('', xy=(x,y), xytext=(0,0), arrowprops=dict(arrowstyle='->', lw=1.1))
    iax.text(x*1.08, y*1.08, label, fontsize=6.2, ha='center', va='center')
iax.axhline(0, lw=.5, alpha=.35); iax.axvline(0, lw=.5, alpha=.35)
iax.set_xlim(-.9,.9); iax.set_ylim(-.9,.9)
iax.set_xticks([]); iax.set_yticks([])
iax.set_title('PCA coefficients', fontsize=7.0, pad=2)

ax = axs[1]
ax.hist(null, bins=38, density=True, alpha=.70)
ax.axvline(obs, linewidth=1.8, label=f'observed = {obs:.3f}')
ax.axvline(exp, linewidth=1.4, linestyle='--', label=f'random = {exp:.3f}')
ax.set_xlabel('Same-subtype fraction among 10 nearest neighbours')
ax.set_ylabel('Permutation density')
ax.set_title('b  Local subtype coherence', loc='left', fontweight='bold')
ax.text(.97,.95, f'ordinary IIb/Ib/Ic/Ic-BL\nsilhouette = {sil:.3f}\npermutation p = {pperm:.4f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8)
ax.legend(frameon=False, fontsize=7, loc='upper left')

save(fig,'Figure1_optical_population')
