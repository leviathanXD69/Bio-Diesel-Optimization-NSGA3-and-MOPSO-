import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler

# ENGINE MOPSO OPTIMIZER - VERSION 5.0 (FULL GPR + VISUALS)
np.random.seed(42)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 140

# ---------------- USER SETTINGS ----------------
filepath = r"C:\Users\User\Desktop\The Second Brain\Bio Diesel Paper\dataaaa\Organized Data.xlsx"

PARTICLES   = 50
ITERATIONS  = 80
ARCHIVE_MAX = 100
save_folder = 'results_mopso_gpr_full'
os.makedirs(save_folder, exist_ok=True)
# =============== LOAD & CLEAN DATA =============
df = pd.read_excel(filepath)
df = df.dropna().reset_index(drop=True)
df = df.rename(columns={'SPEED (RPM)': 'RPM', 'BLEND (%)': 'BLEND'})

obj_cols = ['BSFC', 'BTE', 'NOx', 'CO', 'POWER', 'TORQUE', 'CO2']
points = df[['RPM', 'BLEND']].values

scaler_x = StandardScaler()
points_scaled = scaler_x.fit_transform(points)

def bounds(col):
    return df[col].min(), df[col].max()

rpm_bounds   = bounds('RPM')
blend_bounds = bounds('BLEND')
obj_bounds   = {c: bounds(c) for c in obj_cols}

# TRAIN STABILIZED GPR MODELS
print("\nTraining Stabilized GPR Models... Please wait.")
models = {}
scalers_y = {}

kernel = C(1.0, (1e-5, 1e5)) * RBF([1.0, 1.0], (1e-3, 1e3)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-7, 1e1))

for c in obj_cols:
    y = df[c].values.reshape(-1, 1)
    sy = StandardScaler()
    y_scaled = sy.fit_transform(y)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=1e-5)
    gpr.fit(points_scaled, y_scaled)
    models[c] = gpr
    scalers_y[c] = sy

# STABILIZED GPR ACCURACY REPORT
print("\n" + "="*60)
print("STABILIZED GPR ACCURACY REPORT")
print("="*60)
accuracy_results = []
for c in obj_cols:
    y_true = df[c].values
    y_pred_scaled = models[c].predict(points_scaled).reshape(-1, 1)
    y_pred = scalers_y[c].inverse_transform(y_pred_scaled).flatten()
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy_results.append({'Objective': c, 'R2': r2, 'MAE': mae, 'MAPE%': mape})
print(pd.DataFrame(accuracy_results).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


print("\nExporting GPR validation data for graphing script...")
valid_data = pd.DataFrame()

for c in obj_cols:
    y_true      = df[c].values
    y_pred_s    = models[c].predict(points_scaled).reshape(-1, 1)
    y_pred      = scalers_y[c].inverse_transform(y_pred_s).flatten()
    valid_data[f'{c}_Exp']  = y_true
    valid_data[f'{c}_Pred'] = y_pred

valid_data.to_csv('mopso_validation_data.csv', index=False)
print("Saved: mopso_validation_data.csv")


# CROSS-VALIDATION (LOO + leave-one-blend-out) WITH QUADRATIC RSM BASELINE
def _cv_fit(Xtr, ytr):
    sx = StandardScaler().fit(Xtr)
    sy = StandardScaler().fit(ytr.reshape(-1, 1))
    k = (C(1.0, (1e-5, 1e5)) * RBF([1.0, 1.0], (1e-3, 1e3))
         + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-7, 1e1)))
    g = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=15, alpha=1e-5, random_state=0)
    g.fit(sx.transform(Xtr), sy.transform(ytr.reshape(-1, 1)))
    return g, sx, sy

def _cv_predict(model, Xn):
    g, sx, sy = model
    return sy.inverse_transform(g.predict(sx.transform(Xn)).reshape(-1, 1)).ravel()

def _cv_metrics(y, p):
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return r2, np.mean(np.abs(y - p)), np.mean(np.abs((y - p) / y)) * 100

def run_cross_validation(data, responses, out_prefix='gpr'):
    X = data[['RPM', 'BLEND']].values
    blend = data['BLEND'].values
    n = len(data)
    inner = (blend > blend.min()) & (blend < blend.max())

    r = (data['RPM'].values - 1550) / 650
    b = (blend - 22.5) / 22.5
    A = np.column_stack([np.ones(n), r, b, r * r, b * b, r * b])

    rows, long_frames = [], []
    for c in responses:
        y = data[c].values
        loo, lobo = np.zeros(n), np.zeros(n)
        rsm_loo, rsm_lobo = np.zeros(n), np.zeros(n)

        for i in range(n):
            tr = np.ones(n, bool); tr[i] = False
            loo[i] = _cv_predict(_cv_fit(X[tr], y[tr]), X[i:i+1])[0]
            rsm_loo[i] = A[i] @ np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]

        for bl in np.unique(blend):
            te = blend == bl
            lobo[te] = _cv_predict(_cv_fit(X[~te], y[~te]), X[te])
            rsm_lobo[te] = A[te] @ np.linalg.lstsq(A[~te], y[~te], rcond=None)[0]

        rows.append([c, *_cv_metrics(y, loo),
                        *_cv_metrics(y[inner], lobo[inner]),
                        *_cv_metrics(y, lobo),
                        _cv_metrics(y, rsm_loo)[0],
                        _cv_metrics(y[inner], rsm_lobo[inner])[0]])
        long_frames.append(pd.DataFrame({'response': c, 'RPM': data['RPM'], 'BLEND': data['BLEND'],
                                         'obs': y, 'loo_pred': loo, 'lobo_pred': lobo,
                                         'rsm_loo_pred': rsm_loo, 'rsm_lobo_pred': rsm_lobo}))

    cv = pd.DataFrame(rows, columns=['response',
        'LOO_R2', 'LOO_MAE', 'LOO_MAPE',
        'LOBOinner_R2', 'LOBOinner_MAE', 'LOBOinner_MAPE',
        'LOBOall_R2', 'LOBOall_MAE', 'LOBOall_MAPE',
        'RSM_LOO_R2', 'RSM_LOBOinner_R2'])
    cv.round(4).to_csv(f'{out_prefix}_cv_table.csv', index=False)
    pd.concat(long_frames).to_csv(f'{out_prefix}_cv_predictions.csv', index=False)

    labels = {'BSFC': 'BSFC (kg/kWh)', 'BTE': 'BTE (%)', 'NOx': 'NOₓ (ppm)', 'CO': 'CO (ppm)',
              'POWER': 'POWER (W)', 'TORQUE': 'TORQUE (Nm)', 'CO2': 'CO₂ (% vol)'}
    mae_dec = lambda v: 3 if v < 1 else 2 if v < 10 else 1
    t2 = pd.DataFrame({
        'Response':              [labels.get(c, c) for c in cv.response],
        'GPR LOO R²':            cv.LOO_R2.map('{:.3f}'.format),
        'GPR LOO MAE':           [f'{v:.{mae_dec(v)}f}' for v in cv.LOO_MAE],
        'GPR LOO MAPE (%)':      cv.LOO_MAPE.map('{:.2f}'.format),
        'GPR LOBO R² (B5–B35)':  cv.LOBOinner_R2.map('{:.3f}'.format),
        'GPR LOBO MAPE (%)':     cv.LOBOinner_MAPE.map('{:.1f}'.format),
        'RSM LOO R²':            cv.RSM_LOO_R2.map('{:.3f}'.format),
        'RSM LOBO R² (B5–B35)':  cv.RSM_LOBOinner_R2.map('{:.3f}'.format)})
    t2.to_csv(f'{out_prefix}_table2.csv', index=False, encoding='utf-8-sig')
    print(t2.to_string(index=False))
    print(f"Saved: {out_prefix}_cv_table.csv, {out_prefix}_cv_predictions.csv, {out_prefix}_table2.csv")
    return cv

print("\nRunning LOO / LOBO cross-validation (this takes a few minutes)...")
cv_table = run_cross_validation(df, obj_cols)



# EVALUATION & DESIRABILITY
def evaluate(pos):
    pos_scaled = scaler_x.transform(np.array(pos).reshape(1, -1))
    vals = [scalers_y[c].inverse_transform(models[c].predict(pos_scaled).reshape(-1,1)).item() for c in obj_cols]

    to_maximize = {1, 4, 5}
    norm = []
    for i, col in enumerate(obj_cols):
        lo, hi = obj_bounds[col]
        n = (vals[i] - lo) / (hi - lo) if hi != lo else 0.5
        norm.append(1.0 - n if i in to_maximize else n)
    return np.array(norm), vals

def desirability_min(v, lo, hi):
    if v <= lo: return 1.0
    if v >= hi: return 0.0
    return (hi - v) / (hi - lo)

def desirability_max(v, lo, hi):
    if v >= hi: return 1.0
    if v <= lo: return 0.0
    return (v - lo) / (hi - lo)


weights = np.array([3, 3, 1, 1, 3, 2, 1])

def total_desirability(row):
    epsilon = 0.005
    ds = [
        max(epsilon, desirability_min(row['BSFC'], *obj_bounds['BSFC'])),
        max(epsilon, desirability_max(row['BTE'], *obj_bounds['BTE'])),
        max(epsilon, desirability_min(row['NOx'], *obj_bounds['NOx'])),
        max(epsilon, desirability_min(row['CO'], *obj_bounds['CO'])),
        max(epsilon, desirability_max(row['POWER'], *obj_bounds['POWER'])),
        max(epsilon, desirability_max(row['TORQUE'], *obj_bounds['TORQUE'])),
        max(epsilon, desirability_min(row['CO2'], *obj_bounds['CO2']))
    ]
    prod = np.prod(np.power(ds, weights))
    return prod ** (1.0 / weights.sum())

# MOPSO CORE ENGINE
class Particle:
    def __init__(self):
        self.pos = np.array([np.random.uniform(*rpm_bounds), np.random.uniform(*blend_bounds)])
        self.vel = np.zeros(2)
        self.best_pos = self.pos.copy()
        self.best_obj, _ = evaluate(self.pos)

archive_pos, archive_obj = [], []

def update_archive(pos, obj):
    global archive_pos, archive_obj
    pos_flat = np.array(pos).flatten()
    for p in archive_pos:
        if np.allclose(p, pos_flat, atol=1e-3): return

    dom = []
    reject = False
    for i, o in enumerate(archive_obj):
        if np.all(o <= obj) and np.any(o < obj): reject = True; break
        if np.all(obj <= o) and np.any(obj < o): dom.append(i)

    if reject: return
    for i in sorted(dom, reverse=True):
        del archive_pos[i]; del archive_obj[i]
    archive_pos.append(pos_flat.copy()); archive_obj.append(obj.copy())

    if len(archive_pos) > ARCHIVE_MAX:
        des = [total_desirability({c: evaluate(p)[1][k] for k,c in enumerate(obj_cols)}) for p in archive_pos]
        worst_idx = np.argmin(des)
        del archive_pos[worst_idx]; del archive_obj[worst_idx]

swarm = [Particle() for _ in range(PARTICLES)]
for p in swarm: update_archive(p.pos, p.best_obj)

conv, arch_hist = [], []
print("\nStarting MOPSO Optimization...")

for gen in range(ITERATIONS):
    w = 0.8  # inertia
    for p in swarm:
        leader = archive_pos[np.random.randint(len(archive_pos))]
        r1, r2 = np.random.rand(2), np.random.rand(2)
        p.vel = w*p.vel + 1.5*r1*(p.best_pos-p.pos) + 1.5*r2*(leader-p.pos)
        p.pos = np.clip(p.pos + p.vel, [rpm_bounds[0], blend_bounds[0]], [rpm_bounds[1], blend_bounds[1]])

        o, _ = evaluate(p.pos)
        if np.all(o <= p.best_obj):
            p.best_pos, p.best_obj = p.pos.copy(), o.copy()
        update_archive(p.pos, o)

    arch_hist.append(len(archive_pos))
    cur_des = []
    for p_arch in archive_pos:
        _, vals = evaluate(p_arch)
        cur_des.append(total_desirability({c: vals[k] for k,c in enumerate(obj_cols)}))
    conv.append(np.mean(cur_des))

    if (gen + 1) % 10 == 0:
        print(f"  Iteration {gen+1:>2} | Archive Size: {len(archive_pos):>3} | Mean Desirability: {conv[-1]:.4f}")

# EXPORT & PLOTTING
final_results = []
for p_arch in archive_pos:
    _, vals = evaluate(p_arch)
    row = {'RPM': p_arch[0], 'BLEND': p_arch[1], **{c: vals[i] for i, c in enumerate(obj_cols)}}
    row['Desirability'] = total_desirability(row)
    final_results.append(row)

res = pd.DataFrame(final_results).sort_values('Desirability', ascending=False)
best = res.iloc[0]
res.to_excel('MOPSO_GPR_Full_Results.xlsx', index=False)


res.to_csv('mopso_pareto_results.csv', index=False)
print("Saved: mopso_pareto_results.csv")

conv_df = pd.DataFrame({
    'Iteration':         range(1, ITERATIONS + 1),
    'Mean_Desirability': conv,
    'Archive_Size':      arch_hist
})
conv_df.to_csv('mopso_convergence_data.csv', index=False)
print("Saved: mopso_convergence_data.csv")

print("\nRunning Sensitivity Analysis...")
perturbations = np.linspace(-0.2, 0.2, 25)
sens_targets  = ['BSFC', 'NOx', 'BTE']
colors        = ['red', 'blue', 'green']
sens_rows     = []

plt.figure(figsize=(10, 6))

for color, target in zip(colors, sens_targets):
    changes   = []
    base_val  = best[target]
    t_idx     = obj_cols.index(target)

    for p in perturbations:
        new_rpm       = best['RPM'] * (1 + p)
        _, new_vals   = evaluate([new_rpm, best['BLEND']])
        pct_change    = ((new_vals[t_idx] - base_val) / base_val) * 100
        changes.append(pct_change)
        sens_rows.append({
            'Target':          target,
            'Perturbation_%':  round(p * 100, 2),
            'Change_%':        pct_change
        })

    plt.plot(perturbations * 100, changes,
             label=f'Sensitivity of {target}',
             color=color, marker='o', markersize=4, lw=2)

plt.axhline(0, color='black', lw=1, ls='--')
plt.axvline(0, color='black', lw=1, ls='--')
plt.xlabel('% Deviation from Optimal RPM', fontsize=12)
plt.ylabel('% Change in Objective Value',  fontsize=12)
plt.title('MOPSO: Sensitivity Analysis at Optimal Point', fontsize=14)
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{save_folder}/sensitivity_analysis.png', dpi=300)
print("Saved: sensitivity_analysis.png")

pd.DataFrame(sens_rows).to_csv('mopso_sensitivity_data.csv', index=False)
print("Saved: mopso_sensitivity_data.csv")


# 1. Convergence
plt.figure(); plt.plot(conv, color='darkorange', lw=2); plt.title('MOPSO GPR Convergence'); plt.grid(True)
plt.savefig(f'{save_folder}/convergence.png')

# 2. Archive Size History
plt.figure(); plt.plot(arch_hist, color='navy', lw=2); plt.title('Archive (Pareto) Size History'); plt.grid(True)
plt.savefig(f'{save_folder}/archive_size.png')

# 3. Desirability Surface
rpm_l, blend_l = np.linspace(*rpm_bounds, 40), np.linspace(*blend_bounds, 40)
R, B = np.meshgrid(rpm_l, blend_l); Z = np.zeros_like(R)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        _, vals = evaluate([R[i,j], B[i,j]])
        Z[i,j] = total_desirability({c: vals[k] for k, c in enumerate(obj_cols)})
plt.figure(); plt.contourf(R, B, Z, 20, cmap='magma'); plt.colorbar(label='Desirability')
plt.scatter(res['RPM'], res['BLEND'], c='white', s=20, label='Archive')
plt.scatter(best['RPM'], best['BLEND'], c='cyan', s=100, edgecolors='black', label='Best')
plt.xlabel('RPM'); plt.ylabel('Blend %'); plt.legend(); plt.savefig(f'{save_folder}/surface.png')

# 4. Radar Chart
labels = ['BSFC', 'BTE', 'POWER', 'TORQUE', 'NOx', 'CO2']
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
def get_scaled(row):
    return [(row[l]-obj_bounds[l][0])/(obj_bounds[l][1]-obj_bounds[l][0]) for l in labels] + [(row[labels[0]]-obj_bounds[labels[0]][0])/(obj_bounds[labels[0]][1]-obj_bounds[labels[0]][0])]

plt.figure(figsize=(8,8)); ax = plt.subplot(111, polar=True)
ax.plot(angles, get_scaled(best), 'r-', label='Best'); ax.fill(angles, get_scaled(best), 'r', alpha=0.1)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels); plt.legend(); plt.savefig(f'{save_folder}/radar.png')

# 5. Correlation Heatmap
plt.figure(); sns.heatmap(res[obj_cols + ['Desirability']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.savefig(f'{save_folder}/heatmap.png')

# 6. Pareto Scatter (Archive)
plt.figure(); plt.scatter(res['BSFC'], res['BTE'], c=res['Desirability'], cmap='viridis')
plt.xlabel('BSFC'); plt.ylabel('BTE'); plt.colorbar(label='Desirability'); plt.savefig(f'{save_folder}/scatter.png')

print(f"\nOptimization Complete. Results saved in {save_folder}/ and Excel file generated.")
plt.show()
