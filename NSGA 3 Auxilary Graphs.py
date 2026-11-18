import os

print("Current Working Directory:", os.getcwd())


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
from matplotlib.lines import Line2D
import os

# NSGA-III GRAPHING SCRIPT
# Needs nsga3_pareto/validation/convergence/sensitivity CSVs from the main script
# and gpr_cv_predictions.csv from MOPSO Main Code.py

# ── THEME ──────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size':       12,
    'axes.labelsize':  14,
    'axes.titlesize':  16,
    'figure.dpi':      300,
    'savefig.dpi':     300,
    'font.family':     'serif'
})

save_folder = 'publication_plots_nsga3'
os.makedirs(save_folder, exist_ok=True)

obj_cols = ['BSFC', 'BTE', 'NOx', 'CO', 'POWER', 'TORQUE', 'CO2']


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ── LOAD DATA ──────────────────────────────────────────
try:
    pareto_df = pd.read_csv(os.path.join(BASE_DIR, 'nsga3_pareto_results.csv'))
    valid_df  = pd.read_csv(os.path.join(BASE_DIR, 'nsga3_validation_data.csv'))
    conv_df   = pd.read_csv(os.path.join(BASE_DIR, 'nsga3_convergence_data.csv'))
    sens_df   = pd.read_csv(os.path.join(BASE_DIR, 'nsga3_sensitivity_data.csv'))
    print("All CSV files loaded successfully!\n")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Run the NSGA-III optimisation script first to generate the CSV files.")
    raise SystemExit

CV_CSV = os.path.join(BASE_DIR, 'gpr_cv_predictions.csv')
cv_df = pd.read_csv(CV_CSV) if os.path.exists(CV_CSV) else None
if cv_df is None:
    print("gpr_cv_predictions.csv not found next to this script: LOO/LOBO plots will be skipped.")


# 1. PARITY PLOTS (Experimental vs GPR Predicted)
def plot_parity_grid(df):
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.flatten()

    for i, col in enumerate(obj_cols):
        exp  = df[f'{col}_Exp']
        pred = df[f'{col}_Pred']

        lo = min(exp.min(), pred.min())
        hi = max(exp.max(), pred.max())
        axes[i].plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect fit')

        sns.regplot(x=exp, y=pred, ax=axes[i],
                    scatter_kws={'alpha': 0.6, 'color': 'steelblue', 's': 40},
                    line_kws={'color': 'darkorange', 'lw': 1.5})

        ss_res = np.sum((exp - pred) ** 2)
        ss_tot = np.sum((exp - exp.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        axes[i].annotate(f'R² = {r2:.4f}', xy=(0.05, 0.92),
                         xycoords='axes fraction', fontsize=11,
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

        axes[i].set_title(f'Parity: {col}', fontsize=13)
        axes[i].set_xlabel('Experimental')
        axes[i].set_ylabel('GPR Predicted')

    axes[-1].set_visible(False)
    fig.suptitle('GPR Training Fit — Experimental vs. Predicted (NSGA-III)', fontsize=16, y=1.01)
    plt.tight_layout()
    path = f'{save_folder}/parity_grid.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# 2. RESIDUAL ERROR DISTRIBUTIONS
def plot_residuals(df):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, col in enumerate(obj_cols):
        residuals = df[f'{col}_Exp'] - df[f'{col}_Pred']
        sns.histplot(residuals, kde=True, ax=axes[i],
                     color='mediumpurple', edgecolor='white', bins=12)
        axes[i].axvline(0, color='red', lw=1.5, ls='--')
        axes[i].set_title(f'Residuals: {col}')
        axes[i].set_xlabel('Prediction Error')
        axes[i].set_ylabel('Count')

    axes[-1].set_visible(False)
    fig.suptitle('Residual Error Distribution — NSGA-III GPR Models', fontsize=16, y=1.01)
    plt.tight_layout()
    path = f'{save_folder}/residuals_distribution.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# 2b. LOO / LOBO PARITY PLOTS
BLEND_COLORS = {0: '#222222', 5: '#0072B2', 15: '#009E73', 25: '#E69F00', 35: '#D55E00', 45: '#CC79A7'}

def plot_cv_parity(cv, pred_col, tag, interior_only=False):
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.flatten()
    for i, col in enumerate(obj_cols):
        d = cv[cv.response == col]
        lo = min(d.obs.min(), d[pred_col].min())
        hi = max(d.obs.max(), d[pred_col].max())
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        axes[i].plot([lo, hi], [lo, hi], color='#777777', ls='--', lw=1.3)
        for b, g in d.groupby('BLEND'):
            axes[i].scatter(g.obs, g[pred_col], s=45, c=BLEND_COLORS.get(int(b), 'gray'),
                            edgecolors='white', linewidths=0.4, zorder=3)
        m = d[(d.BLEND > d.BLEND.min()) & (d.BLEND < d.BLEND.max())] if interior_only else d
        r2 = 1 - np.sum((m.obs - m[pred_col]) ** 2) / np.sum((m.obs - m.obs.mean()) ** 2)
        mape = np.mean(np.abs((m.obs - m[pred_col]) / m.obs)) * 100
        axes[i].text(0.96, 0.05, f'R² = {r2:.3f}\nMAPE = {mape:.2f} %',
                     transform=axes[i].transAxes, ha='right', va='bottom', fontsize=11,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCCCCC'))
        axes[i].set_xlim(lo, hi)
        axes[i].set_ylim(lo, hi)
        axes[i].set_title(f'{tag} Parity: {col}', fontsize=13)
        axes[i].set_xlabel('Measured')
        axes[i].set_ylabel('Predicted')

    handles = [Line2D([], [], marker='o', ls='', mfc=c, mec='white', label=f'B{b}')
               for b, c in BLEND_COLORS.items()]
    axes[-1].axis('off')
    axes[-1].legend(handles=handles, loc='center', title='Biodiesel blend', frameon=False)
    note = ' (metrics on interior blends B5–B35)' if interior_only else ''
    fig.suptitle(f'GPR {tag} Cross-Validation — Measured vs. Predicted{note}', fontsize=16, y=1.01)
    plt.tight_layout()
    path = f'{save_folder}/parity_{tag.lower()}.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# 2c. LOO / LOBO RESIDUAL DISTRIBUTIONS
def plot_cv_residuals(cv, pred_col, tag):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    for i, col in enumerate(obj_cols):
        d = cv[cv.response == col]
        r = d.obs - d[pred_col]
        sns.histplot(r, bins=14, ax=axes[i], color='#7B8FA6', edgecolor='white')
        axes[i].axvline(0, color='#D55E00', ls='--', lw=1.5)
        axes[i].text(0.97, 0.95, f'mean = {r.mean():.3g}\nSD = {r.std(ddof=1):.3g}',
                     transform=axes[i].transAxes, ha='right', va='top', fontsize=11,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCCCCC'))
        axes[i].set_title(f'{tag} Residuals: {col}')
        axes[i].set_xlabel('Measured − predicted')
        axes[i].set_ylabel('Count')
    axes[-1].set_visible(False)
    fig.suptitle(f'GPR {tag} Residual Distributions', fontsize=16, y=1.01)
    plt.tight_layout()
    path = f'{save_folder}/residuals_{tag.lower()}.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# 3. CONVERGENCE + HYPERVOLUME (two-panel)
def plot_convergence(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(df['Generation'], df['Mean_Desirability'],
             color='darkorange', lw=2.5, marker='o', markersize=3)
    ax1.set_title('NSGA-III Convergence — Mean Desirability')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean Desirability of Pareto Front')
    ax1.grid(alpha=0.4)

    ax2.plot(df['Generation'], df['Pareto_Size'],
             color='steelblue', lw=2.5, marker='s', markersize=3)
    ax2.set_title('Pareto Front Growth Over Generations')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Pareto Front Size')
    ax2.grid(alpha=0.4)

    plt.tight_layout()
    path = f'{save_folder}/convergence_and_pareto_size.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# 4. SENSITIVITY ANALYSIS REPLOT
def plot_sensitivity(df):
    plt.figure(figsize=(10, 6))
    colors  = {'BSFC': 'red', 'NOx': 'blue', 'BTE': 'green'}
    targets = df['Target'].unique()

    for target in targets:
        sub = df[df['Target'] == target].sort_values('Perturbation_%')
        plt.plot(sub['Perturbation_%'], sub['Change_%'],
                 label=f'Sensitivity of {target}',
                 color=colors.get(target, 'gray'),
                 marker='o', markersize=4, lw=2)

    plt.axhline(0, color='black', lw=1, ls='--')
    plt.axvline(0, color='black', lw=1, ls='--')
    plt.xlabel('% Deviation from Optimal RPM')
    plt.ylabel('% Change in Objective Value')
    plt.title('NSGA-III: System Sensitivity Analysis at Optimal Point')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = f'{save_folder}/sensitivity_analysis.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# 5. RADAR CHART (Best vs Pareto Mean)
def plot_radar_comparison(df):
    labels   = ['BSFC', 'BTE', 'POWER', 'TORQUE', 'NOx', 'CO2']
    df_norm  = (df[labels] - df[labels].min()) / (df[labels].max() - df[labels].min())
    df_norm  = df_norm.fillna(0.5)

    best_idx  = df['Desirability'].idxmax()
    best_vals = df_norm.loc[best_idx].values
    mean_vals = df_norm.mean().values

    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    best_vals = np.concatenate((best_vals, [best_vals[0]]))
    mean_vals = np.concatenate((mean_vals, [mean_vals[0]]))
    angles    = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, best_vals, color='blue',   alpha=0.20)
    ax.plot(angles, best_vals, color='blue',   lw=2,  label='Best Solution')
    ax.fill(angles, mean_vals, color='orange', alpha=0.15)
    ax.plot(angles, mean_vals, color='orange', lw=2,  label='Pareto Mean', ls='--')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels([])
    ax.set_title('Performance Fingerprint (Normalized)\nNSGA-III Best vs Pareto Mean',
                 pad=20, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    path = f'{save_folder}/radar_comparison.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# 6. PARETO SCATTER MATRIX (BSFC / BTE / NOx / CO2)
def plot_pareto_scatter_matrix(df):
    pairs = [
        ('BSFC', 'BTE',   'BSFC vs BTE'),
        ('NOx',  'CO2',   'NOx vs CO2'),
        ('BSFC', 'NOx',   'BSFC vs NOx'),
        ('BTE',  'POWER', 'BTE vs POWER'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (xcol, ycol, title) in zip(axes, pairs):
        sc = ax.scatter(df[xcol], df[ycol],
                        c=df['Desirability'], cmap='plasma',
                        edgecolors='k', lw=0.3, s=50, alpha=0.85)
        plt.colorbar(sc, ax=ax, label='Desirability')
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(title)

    fig.suptitle('NSGA-III Pareto Front — Trade-off Slices', fontsize=16)
    plt.tight_layout()
    path = f'{save_folder}/pareto_scatter_matrix.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# 7. CORRELATION HEATMAP
def plot_correlation(df):
    cols = ['RPM', 'BLEND'] + obj_cols + ['Desirability']
    plt.figure(figsize=(11, 9))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',
                linewidths=0.5, annot_kws={'size': 9})
    plt.title('Correlation Matrix — NSGA-III Pareto Front\n(Inputs + Objectives)',
              fontsize=14)
    plt.tight_layout()
    path = f'{save_folder}/correlation_heatmap.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# 8. PARALLEL COORDINATES PLOT
def plot_parallel_coordinates(df):
    df_norm = df[obj_cols + ['Desirability']].copy()
    for col in df_norm.columns:
        lo, hi = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = (df_norm[col] - lo) / (hi - lo + 1e-12)

    df_norm['D_rank'] = pd.cut(df['Desirability'], bins=5,
                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    fig, ax = plt.subplots(figsize=(16, 7))
    palette = {'Very Low': '#d73027', 'Low': '#fc8d59',
               'Medium':   '#fee090', 'High': '#91bfdb', 'Very High': '#4575b4'}

    cols_to_plot = obj_cols + ['Desirability']
    x_positions  = list(range(len(cols_to_plot)))

    for _, row in df_norm.iterrows():
        color = palette.get(str(row['D_rank']), 'grey')
        ax.plot(x_positions, [row[c] for c in cols_to_plot],
                color=color, alpha=0.4, lw=1)

    patches = [mpatches.Patch(color=v, label=k) for k, v in palette.items()]
    ax.legend(handles=patches, title='Desirability', loc='upper right', fontsize=10)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(cols_to_plot, fontsize=11, rotation=15)
    ax.set_ylabel('Normalized Value')
    ax.set_title('Parallel Coordinates — NSGA-III Pareto Front\n(coloured by Desirability tier)',
                 fontsize=14)
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    path = f'{save_folder}/parallel_coordinates.png'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")


# 9. INTERACTIVE 3-D PARETO FRONT (Plotly)
def plot_3d_pareto(df):
    fig = px.scatter_3d(
        df,
        x='BSFC', y='NOx', z='BTE',
        color='Desirability',
        color_continuous_scale='Plasma',
        size='POWER',
        hover_data=['RPM', 'BLEND', 'CO2', 'TORQUE'],
        title='NSGA-III — Interactive 3D Pareto Front (BSFC / NOx / BTE)'
    )
    fig.update_layout(scene=dict(
        xaxis_title='BSFC',
        yaxis_title='NOx',
        zaxis_title='BTE'
    ))
    path = f'{save_folder}/pareto_3d_interactive.html'
    fig.write_html(path)
    print(f"Saved: {path}  (open in any browser to rotate)")


# 10. BEST SOLUTION SUMMARY TABLE
def print_best_solutions(df, top_n=5):
    top = df.nlargest(top_n, 'Desirability')[['RPM', 'BLEND'] + obj_cols + ['Desirability']]
    print("\n" + "=" * 70)
    print(f"TOP {top_n} NSGA-III SOLUTIONS (ranked by Desirability)")
    print("=" * 70)
    print(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 70)

    top.to_csv(f'{save_folder}/top_{top_n}_solutions.csv', index=False)
    print(f"Saved: {save_folder}/top_{top_n}_solutions.csv\n")


# EXECUTE ALL PLOTS
print_best_solutions(pareto_df)
plot_parity_grid(valid_df)
plot_residuals(valid_df)
if cv_df is not None:
    plot_cv_parity(cv_df, 'loo_pred', 'LOO')
    plot_cv_residuals(cv_df, 'loo_pred', 'LOO')
    plot_cv_parity(cv_df, 'lobo_pred', 'LOBO', interior_only=True)
    plot_cv_residuals(cv_df, 'lobo_pred', 'LOBO')
plot_convergence(conv_df)
plot_sensitivity(sens_df)
plot_radar_comparison(pareto_df)
plot_pareto_scatter_matrix(pareto_df)
plot_correlation(pareto_df)
plot_parallel_coordinates(pareto_df)
plot_3d_pareto(pareto_df)

print(f"\nAll plots saved in '{save_folder}/'")
plt.show()


import platform

# Automatically open the output folder when script finishes
if platform.system() == 'Windows':
    os.startfile(save_folder)
elif platform.system() == 'Darwin':
    os.system(f'open "{save_folder}"')
else:
    os.system(f'xdg-open "{save_folder}"')
