import os

print("\n" + "="*60)
print("CURRENT WORKING DIRECTORY (Where your CSVs & Excel go):")
print(os.getcwd())
print("\nFIGURES FOLDER LOCATION:")
print(os.path.abspath('results_figures_nsga3'))
print("="*60 + "\n")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler

# ENGINE NSGA-III OPTIMIZER - VERSION 1.0
np.random.seed(42)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 140

# ---------------- USER SETTINGS ----------------
filepath = r"C:\Users\User\Desktop\The Second Brain\Bio Diesel Paper\dataaaa\Organized Data.xlsx"

N_OBJ       = 7
N_DIVISIONS = 3        # 84 reference points for 7 objectives (POP_SIZE >= 84)
POP_SIZE    = 84
GENERATIONS = 60
SBX_ETA     = 15
PM_ETA      = 20
CROSSOVER_P = 0.9
MUTATION_P  = 0.1
save_folder = 'results_figures_nsga3'
os.makedirs(save_folder, exist_ok=True)
# =============== LOAD & CLEAN DATA =============
df = pd.read_excel(filepath)
df = df.dropna().reset_index(drop=True)
df = df.rename(columns={'SPEED (RPM)': 'RPM', 'BLEND (%)': 'BLEND'})

obj_cols = ['BSFC', 'BTE', 'NOx', 'CO', 'POWER', 'TORQUE', 'CO2']

TO_MAXIMIZE = {1, 4, 5}   # BTE, POWER, TORQUE (negated internally)

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
models, scalers_y = {}, {}

kernel = (C(1.0, (1e-5, 1e5))
          * RBF([1.0, 1.0], (1e-3, 1e3))
          + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-7, 1e1)))

for c in obj_cols:
    y = df[c].values.reshape(-1, 1)
    sy = StandardScaler()
    y_scaled = sy.fit_transform(y)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=1e-5)
    gpr.fit(points_scaled, y_scaled)
    models[c] = gpr
    scalers_y[c] = sy

# GPR ACCURACY REPORT
print("\n" + "="*60)
print("STABILIZED GPR ACCURACY REPORT")
print("="*60)
accuracy_results = []
for c in obj_cols:
    y_true = df[c].values
    y_pred_scaled = models[c].predict(points_scaled).reshape(-1, 1)
    y_pred = scalers_y[c].inverse_transform(y_pred_scaled).flatten()
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy_results.append({'Objective': c, 'R2': r2, 'MAE': mae, 'MAPE%': mape})
print(pd.DataFrame(accuracy_results).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# GPR EVALUATION
def evaluate_raw(pos):
    """Return raw (physical-unit) objective values for a candidate [RPM, BLEND]."""
    pos_s = scaler_x.transform(np.array(pos).reshape(1, -1))
    vals = [
        scalers_y[c].inverse_transform(
            models[c].predict(pos_s).reshape(-1, 1)
        ).item()
        for c in obj_cols
    ]
    return np.array(vals, dtype=float)

print("\nGenerating Validation Data for separate Visualizer...")
valid_data = pd.DataFrame()

for c in obj_cols:
    y_true = df[c].values

    y_pred_scaled = models[c].predict(points_scaled).reshape(-1, 1)
    y_pred = scalers_y[c].inverse_transform(y_pred_scaled).flatten()

    valid_data[f'{c}_Exp'] = y_true
    valid_data[f'{c}_Pred'] = y_pred

# Save to CSV so the second script can access it
valid_data.to_csv('nsga3_validation_data.csv', index=False)
print("Validation CSV saved successfully for parity plots.")


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



def normalize_for_nsga3(vals):
    """
    Convert raw objectives to a minimisation vector.
    Maximised objectives are negated so ALL objectives are minimised.
    """
    v = vals.copy()
    for i in TO_MAXIMIZE:
        v[i] = -v[i]
    return v

# DESIRABILITY (unchanged from NSGA-II version)
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
        max(epsilon, desirability_min(row['BSFC'],  *obj_bounds['BSFC'])),
        max(epsilon, desirability_max(row['BTE'],   *obj_bounds['BTE'])),
        max(epsilon, desirability_min(row['NOx'],   *obj_bounds['NOx'])),
        max(epsilon, desirability_min(row['CO'],    *obj_bounds['CO'])),
        max(epsilon, desirability_max(row['POWER'], *obj_bounds['POWER'])),
        max(epsilon, desirability_max(row['TORQUE'],*obj_bounds['TORQUE'])),
        max(epsilon, desirability_min(row['CO2'],   *obj_bounds['CO2']))
    ]
    prod = np.prod(np.power(ds, weights))
    return prod ** (1.0 / weights.sum())

# NSGA-III CORE — Reference Points (Das-Dennis method)
def _recursive_ref(n_obj, n_div, current, remaining, ref_pts):
    """Recursively enumerate Das-Dennis reference point coordinates."""
    if n_obj == 1:
        ref_pts.append(current + [remaining / n_div])
        return
    for i in range(remaining + 1):
        _recursive_ref(n_obj - 1, n_div, current + [i / n_div],
                       remaining - i, ref_pts)

def generate_reference_points(n_obj, n_div):
    """
    Generate Das-Dennis structured reference points on the unit simplex.
    Number of points = C(n_obj + n_div - 1, n_div).
    """
    ref_pts = []
    _recursive_ref(n_obj, n_div, [], n_div, ref_pts)
    return np.array(ref_pts)

REF_POINTS = generate_reference_points(N_OBJ, N_DIVISIONS)
print(f"\nNSGA-III Reference Points generated: {len(REF_POINTS)}")
print(f"Population size: {POP_SIZE}")

# NSGA-III CORE — Non-dominated Sort (same as NSGA-II)
def fast_non_dominated_sort(F):
    n = len(F)
    S, n_dom, fronts = [[] for _ in range(n)], np.zeros(n, dtype=int), [[]]
    for i in range(n):
        for j in range(i + 1, n):
            Fi, Fj = F[i], F[j]
            if np.all(Fi <= Fj) and np.any(Fi < Fj):
                S[i].append(j); n_dom[j] += 1
            elif np.all(Fj <= Fi) and np.any(Fj < Fi):
                S[j].append(i); n_dom[i] += 1
        if n_dom[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    nxt.append(j)
        k += 1; fronts.append(nxt)
    return [f for f in fronts if f]

# NSGA-III CORE — Normalisation & Association
def get_ideal_nadir(F):
    """Compute ideal (min per objective) and nadir (max per objective)."""
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    return ideal, nadir

def normalize_population(F, ideal, nadir):
    """Translate & scale so each objective lies in [0, ∞), with front on unit simplex."""
    span = nadir - ideal
    span[span < 1e-12] = 1e-12
    return (F - ideal) / span

def associate_to_reference(F_norm, ref_pts):
    """
    For each solution find the closest reference line (from origin).
    Returns:
        assoc  : (n_solutions,) index into ref_pts
        dist   : (n_solutions,) perpendicular distance to assigned ref line
    """
    ref_norm = ref_pts / (np.linalg.norm(ref_pts, axis=1, keepdims=True) + 1e-12)

    assoc = np.empty(len(F_norm), dtype=int)
    dist  = np.empty(len(F_norm), dtype=float)

    for i, f in enumerate(F_norm):
        scalar_proj = ref_norm @ f
        scalar_proj = np.maximum(scalar_proj, 0.0)
        proj_pts = scalar_proj[:, None] * ref_norm
        dists = np.linalg.norm(f - proj_pts, axis=1)
        best = np.argmin(dists)
        assoc[i] = best
        dist[i]  = dists[best]

    return assoc, dist

# NSGA-III CORE — Niching Selection
def niching_selection(candidates, needed, niche_count, assoc, dist):
    """
    Select `needed` solutions from `candidates` using NSGA-III niche preservation.
    candidates : list of global indices in the last critical front
    niche_count: dict {ref_idx: count already chosen from prev fronts}
    assoc      : global assoc array (n_combined,)
    dist       : global dist  array (n_combined,)
    """
    selected = []
    niche_count = niche_count.copy()

    while len(selected) < needed:
        j_min = min(niche_count.values())
        ref_with_min = [r for r, c in niche_count.items()
                        if c == j_min
                        and any(assoc[s] == r for s in candidates)]
        if not ref_with_min:
            ref_with_min = list(niche_count.keys())

        chosen_ref = ref_with_min[np.random.randint(len(ref_with_min))]

        cands_in_ref = [s for s in candidates if assoc[s] == chosen_ref]
        if not cands_in_ref:
            niche_count[chosen_ref] += 1
            continue

        if niche_count[chosen_ref] == 0:
            pick = min(cands_in_ref, key=lambda s: dist[s])
        else:
            pick = cands_in_ref[np.random.randint(len(cands_in_ref))]

        selected.append(pick)
        candidates.remove(pick)
        niche_count[chosen_ref] = niche_count.get(chosen_ref, 0) + 1

    return selected

# NSGA-III CORE — Genetic Operators (same as NSGA-II)
def tournament_select_nsga3(pop_size, ranks, assoc, dist, niche_count):
    """Binary tournament: rank first, then niche count, then distance."""
    c = np.random.choice(pop_size, 2, replace=False)
    a, b = c[0], c[1]
    if ranks[a] < ranks[b]:
        return a
    if ranks[b] < ranks[a]:
        return b
    na, nb = niche_count.get(assoc[a], 0), niche_count.get(assoc[b], 0)
    if na < nb: return a
    if nb < na: return b
    return a if dist[a] <= dist[b] else b

def sbx_crossover(p1, p2, lb, ub):
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if np.random.rand() < 0.5 and abs(p1[i] - p2[i]) > 1e-9:
            y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
            u = np.random.rand()
            if (y1 - lb[i]) < (ub[i] - y2):
                beta = 1.0 + (2.0 * (y1 - lb[i]) / (y2 - y1))
            else:
                beta = 1.0 + (2.0 * (ub[i] - y2) / (y2 - y1))
            alpha = 2.0 - beta ** (-(SBX_ETA + 1))
            bq = ((u * alpha) ** (1 / (SBX_ETA + 1)) if u <= 1 / alpha
                  else (1 / (2.0 - u * alpha)) ** (1 / (SBX_ETA + 1)))
            c1[i] = 0.5 * ((y1 + y2) - bq * (y2 - y1))
            c2[i] = 0.5 * ((y1 + y2) + bq * (y2 - y1))
    return np.clip(c1, lb, ub), np.clip(c2, lb, ub)

def polynomial_mutation(x, lb, ub):
    child = x.copy()
    for i in range(len(x)):
        if np.random.rand() < MUTATION_P:
            u = np.random.rand()
            d1 = (x[i] - lb[i]) / (ub[i] - lb[i] + 1e-12)
            d2 = (ub[i] - x[i]) / (ub[i] - lb[i] + 1e-12)
            dq = ((2*u + (1 - 2*u)*(1 - d1)**(PM_ETA+1))**(1/(PM_ETA+1)) - 1
                  if u < 0.5 else
                  1 - (2*(1-u) + 2*(u-0.5)*(1-d2)**(PM_ETA+1))**(1/(PM_ETA+1)))
            child[i] = np.clip(x[i] + dq * (ub[i] - lb[i]), lb[i], ub[i])
    return child

# NSGA-III MAIN LOOP
lb = np.array([rpm_bounds[0],   blend_bounds[0]])
ub = np.array([rpm_bounds[1],   blend_bounds[1]])

pop   = np.column_stack([np.random.uniform(lb[0], ub[0], POP_SIZE),
                          np.random.uniform(lb[1], ub[1], POP_SIZE)])

F_pop = np.array([normalize_for_nsga3(evaluate_raw(p)) for p in pop])

conv, arch_hist = [], []
print("\nStarting NSGA-III GPR Optimisation...")

for gen in range(GENERATIONS):
    fronts = fast_non_dominated_sort(F_pop)
    ranks  = np.zeros(POP_SIZE, dtype=int)
    for r, f in enumerate(fronts):
        for idx in f:
            ranks[idx] = r

    ideal, nadir = get_ideal_nadir(F_pop)
    F_norm       = normalize_population(F_pop, ideal, nadir)
    assoc, dist  = associate_to_reference(F_norm, REF_POINTS)

    niche_count = {i: 0 for i in range(len(REF_POINTS))}
    for idx in range(POP_SIZE):
        niche_count[assoc[idx]] = niche_count.get(assoc[idx], 0) + 1

    off, F_off = np.empty_like(pop), np.empty_like(F_pop)
    for k in range(0, POP_SIZE, 2):
        p1 = tournament_select_nsga3(POP_SIZE, ranks, assoc, dist, niche_count)
        p2 = tournament_select_nsga3(POP_SIZE, ranks, assoc, dist, niche_count)
        if np.random.rand() < CROSSOVER_P:
            c1, c2 = sbx_crossover(pop[p1], pop[p2], lb, ub)
        else:
            c1, c2 = pop[p1].copy(), pop[p2].copy()
        off[k],   F_off[k]   = polynomial_mutation(c1, lb, ub), None
        off[k+1], F_off[k+1] = polynomial_mutation(c2, lb, ub), None

    for k in range(POP_SIZE):
        F_off[k] = normalize_for_nsga3(evaluate_raw(off[k]))

    comb   = np.vstack([pop, off])
    F_comb = np.vstack([F_pop, F_off])
    n_comb = len(comb)

    f_c    = fast_non_dominated_sort(F_comb)

    ideal_c, nadir_c = get_ideal_nadir(F_comb)
    F_norm_c         = normalize_population(F_comb, ideal_c, nadir_c)
    assoc_c, dist_c  = associate_to_reference(F_norm_c, REF_POINTS)

    new_idx     = []
    niche_sel   = {i: 0 for i in range(len(REF_POINTS))}

    for front in f_c:
        if len(new_idx) + len(front) <= POP_SIZE:
            new_idx.extend(front)
            for idx in front:
                niche_sel[assoc_c[idx]] = niche_sel.get(assoc_c[idx], 0) + 1
        else:
            needed    = POP_SIZE - len(new_idx)
            remaining = list(front)
            chosen    = niching_selection(remaining, needed,
                                          niche_sel, assoc_c, dist_c)
            new_idx.extend(chosen)
            break

    pop   = comb[new_idx]
    F_pop = F_comb[new_idx]

    pf    = fast_non_dominated_sort(F_pop)[0]
    arch_hist.append(len(pf))
    p_des = [
        total_desirability({c: evaluate_raw(pop[idx])[i]
                            for i, c in enumerate(obj_cols)})
        for idx in pf
    ]
    conv.append(np.mean(p_des))
    if (gen + 1) % 10 == 0:
        print(f"  Gen {gen+1:>2} | Pareto: {len(pf):>3} | Mean Des: {conv[-1]:.4f}")


conv_df = pd.DataFrame({
    'Generation':        range(1, GENERATIONS + 1),
    'Mean_Desirability': conv,
    'Pareto_Size':       arch_hist
})
conv_df.to_csv('nsga3_convergence_data.csv', index=False)
print("Convergence CSV saved.")


# EXPORT RESULTS
final_pf = fast_non_dominated_sort(F_pop)[0]
final_results = []
for idx in final_pf:
    raw = evaluate_raw(pop[idx])
    row = {'RPM': pop[idx][0], 'BLEND': pop[idx][1],
           **{c: raw[i] for i, c in enumerate(obj_cols)}}
    row['Desirability'] = total_desirability(row)
    final_results.append(row)

res  = pd.DataFrame(final_results).sort_values('Desirability', ascending=False)
best = res.iloc[0]
res.to_excel('NSGA3_GPR_Full_Results.xlsx', index=False)


print("\nRunning Perturbation Analysis for Journal Visuals...")
perturbations = np.linspace(-0.2, 0.2, 25)
sens_targets = ['BSFC', 'NOx', 'BTE']
colors = ['red', 'blue', 'green']

plt.figure(figsize=(10, 6))

for i, target in enumerate(sens_targets):
    changes = []
    base_val = best[target]
    t_idx = obj_cols.index(target)

    for p in perturbations:
        new_rpm = best['RPM'] * (1 + p)
        new_vals = evaluate_raw([new_rpm, best['BLEND']])
        new_val = new_vals[t_idx]

        changes.append(((new_val - base_val) / base_val) * 100)

    plt.plot(perturbations * 100, changes, label=f'Sensitivity of {target}',
             color=colors[i], marker='o', markersize=4, lw=2)

plt.axhline(0, color='black', lw=1, ls='--')
plt.axvline(0, color='black', lw=1, ls='--')
plt.xlabel('% Deviation from Optimal RPM', fontsize=12)
plt.ylabel('% Change in Objective Value', fontsize=12)
plt.title('System Sensitivity Analysis at Optimal Point', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

sens_rows = []
for target in sens_targets:
    base_val = best[target]
    t_idx = obj_cols.index(target)
    for p in perturbations:
        new_vals = evaluate_raw([best['RPM'] * (1 + p), best['BLEND']])
        pct_change = ((new_vals[t_idx] - base_val) / base_val) * 100
        sens_rows.append({'Target': target, 'Perturbation_%': p * 100, 'Change_%': pct_change})
pd.DataFrame(sens_rows).to_csv('nsga3_sensitivity_data.csv', index=False)
print("Sensitivity CSV saved.")

plt.savefig(f'{save_folder}/sensitivity_analysis_fixed.png', dpi=300)

print("\n" + "="*60)
print("BEST SOLUTION (Highest Desirability)")
print("="*60)
print(best.to_string())

# PLOTS
# 1. Convergence
plt.figure()
plt.plot(conv, color='green', lw=2)
plt.title('NSGA-III GPR Convergence (Mean Desirability)')
plt.xlabel('Generation'); plt.ylabel('Mean Desirability')
plt.grid(True)
plt.savefig(f'{save_folder}/convergence.png', bbox_inches='tight')

# 2. Pareto Front Size History
plt.figure()
plt.plot(arch_hist, color='steelblue', lw=2)
plt.title('Pareto Front Size — NSGA-III')
plt.xlabel('Generation'); plt.ylabel('Pareto Front Size')
plt.grid(True)
plt.savefig(f'{save_folder}/pareto_size.png', bbox_inches='tight')

# 3. Desirability Surface
rpm_l   = np.linspace(*rpm_bounds,   40)
blend_l = np.linspace(*blend_bounds, 40)
R, B    = np.meshgrid(rpm_l, blend_l)
Z       = np.zeros_like(R)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        raw    = evaluate_raw([R[i, j], B[i, j]])
        Z[i,j] = total_desirability({c: raw[k] for k, c in enumerate(obj_cols)})
plt.figure()
plt.contourf(R, B, Z, 20, cmap='viridis')
plt.colorbar(label='Desirability')
plt.scatter(res['RPM'], res['BLEND'], c='white', s=20, label='Pareto Front')
plt.scatter(best['RPM'], best['BLEND'], c='red', s=100,
            edgecolors='white', zorder=5, label='Best')
plt.xlabel('RPM'); plt.ylabel('Blend %')
plt.title('Desirability Surface — NSGA-III')
plt.legend()
plt.savefig(f'{save_folder}/surface.png', bbox_inches='tight')

# 4. Radar Chart
labels  = ['BSFC', 'BTE', 'POWER', 'TORQUE', 'NOx', 'CO2']
angles  = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]

def get_scaled(row):
    vals = [(row[l] - obj_bounds[l][0]) / (obj_bounds[l][1] - obj_bounds[l][0])
            for l in labels]
    return vals + [vals[0]]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles, get_scaled(best), 'b-', lw=2, label='Best Solution')
ax.fill(angles, get_scaled(best), 'b', alpha=0.1)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=12)
plt.title('Best Solution — Objective Radar', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.savefig(f'{save_folder}/radar.png', bbox_inches='tight')

# 5. Correlation Heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(res[obj_cols + ['Desirability']].corr(),
            annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Objective Correlation Heatmap — NSGA-III Pareto Front')
plt.savefig(f'{save_folder}/heatmap.png', bbox_inches='tight')

# 6. Pareto Scatter (BSFC vs BTE)
plt.figure(); plt.scatter(res['BSFC'], res['BTE'], c=res['Desirability'], cmap='viridis')
plt.xlabel('BSFC'); plt.ylabel('BTE'); plt.colorbar(label='Desirability'); plt.savefig(f'{save_folder}/scatter.png')

# 7. Reference Point Association Plot (NEW — unique to NSGA-III)
ideal_f, nadir_f = get_ideal_nadir(F_pop)
F_norm_f         = normalize_population(F_pop, ideal_f, nadir_f)
assoc_f, dist_f  = associate_to_reference(F_norm_f, REF_POINTS)

plt.figure()
niche_usage = np.bincount(assoc_f, minlength=len(REF_POINTS))
plt.bar(range(len(REF_POINTS)), niche_usage, color='steelblue', alpha=0.8)
plt.xlabel('Reference Point Index'); plt.ylabel('Solutions Assigned')
plt.title('Reference Point Niche Utilisation — Final Generation')
plt.grid(axis='y', alpha=0.4)
plt.savefig(f'{save_folder}/niche_utilisation.png', bbox_inches='tight')

print(f"\nAll charts saved in '{save_folder}/'")
plt.show()

res.to_csv('nsga3_pareto_results.csv', index=False)
print("Data saved for Visualization script.")


# FOUR-OBJECTIVE PARETO ANALYSIS: TABLE 10, TABLE 11, FIGURE 15
# Self-contained: fits its own GPRs and runs its own 4-objective MOPSO and NSGA-III (10 seeds each).
# Needs pymoo:  pip install pymoo
TESTED_POINTS = {'MOPSO point': (1899.923, 13.35751), 'NSGA-III point': (1899.991, 12.11478)}

def run_four_objective_analysis(data, out_dir, seeds=10):
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.indicators.hv import HV
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from matplotlib.lines import Line2D

    d4 = data.copy()
    wC = (1 - d4.BLEND / 100) * 0.865 + (d4.BLEND / 100) * 0.770
    nd_mol = (wC * 1000 / 12.011) / (d4.CO2 / 100 + d4.CO * 1e-6)
    d4['NOx_sp'] = d4.NOx * 1e-6 * nd_mol * 46.0 * d4.BSFC
    d4['CO_sp'] = d4.CO * 1e-6 * nd_mol * 28.0 * d4.BSFC

    OBJ = ['BSFC', 'POWER', 'CO_sp', 'NOx_sp']
    MAXI = np.array([False, True, False, False])
    X = d4[['RPM', 'BLEND']].values
    models = {c: _cv_fit(X, d4[c].values) for c in OBJ}

    def F(Xn):
        v = np.column_stack([_cv_predict(models[c], Xn) for c in OBJ])
        return np.where(MAXI, -v, v)

    lb, ub = np.array([900., 0.]), np.array([2200., 45.])
    grid = np.array([[r, b] for r in np.arange(900, 2200.1, 5) for b in np.arange(0, 45.01, 0.25)])
    Fg = F(grid)
    g_idx = NonDominatedSorting().do(Fg, only_non_dominated_front=True)
    Pg, Xg_front = Fg[g_idx], grid[g_idx]
    ideal, nadir = Fg.min(0), Fg.max(0)
    norm = lambda Fx: (Fx - ideal) / (nadir - ideal)
    hv = HV(ref_point=np.ones(4) * 1.1)
    hv_grid = hv(norm(Pg))
    print(f"Grid front: {len(Pg)} non-dominated points, HV = {hv_grid:.4f}")

    lo, hi = d4[OBJ].min().values, d4[OBJ].max().values
    def desir(vals, w):
        w = np.asarray(w, float)
        dd = np.clip(np.where(MAXI, (vals - lo) / (hi - lo), (hi - vals) / (hi - lo)), 0.005, 1)
        return np.prod(dd ** w, axis=1) ** (1 / w.sum())

    class Prob(Problem):
        def __init__(s): super().__init__(n_var=2, n_obj=4, xl=lb, xu=ub)
        def _evaluate(s, x, out, *a, **k): out['F'] = F(x)

    def run_nsga3(seed):
        rd = get_reference_directions('das-dennis', 4, n_partitions=6)
        r = minimize(Prob(), NSGA3(pop_size=84, ref_dirs=rd), ('n_gen', 60), seed=seed, verbose=False)
        return r.X, r.F

    def run_mopso(seed, n=50, it=80, amax=100):
        rng = np.random.default_rng(seed)
        w_, c1, c2 = 0.7298, 1.49618, 1.49618
        pos = rng.uniform(lb, ub, (n, 2)); vel = np.zeros((n, 2))
        pb = pos.copy(); pbF = F(pos)
        A, AF = np.empty((0, 2)), np.empty((0, 4))

        def crowd(AF, edge):
            N = norm(AF); o = np.argsort(N, axis=0); cd = np.zeros(len(AF))
            for j in range(4):
                s = o[:, j]; cd[s[0]] = cd[s[-1]] = edge
                if len(AF) > 2: cd[s[1:-1]] += N[s[2:], j] - N[s[:-2], j]
            return cd

        def upd(A, AF, Xn, Fn):
            A, AF = np.vstack([A, Xn]), np.vstack([AF, Fn])
            k = NonDominatedSorting().do(AF, only_non_dominated_front=True)
            A, AF = A[k], AF[k]
            while len(A) > amax:
                dmin = np.argmin(crowd(AF, np.inf))
                A, AF = np.delete(A, dmin, 0), np.delete(AF, dmin, 0)
            return A, AF

        A, AF = upd(A, AF, pos, pbF)
        for _ in range(it):
            cd = crowd(AF, 1e3)
            ld = A[rng.choice(len(A), n, p=cd / cd.sum() if cd.sum() > 0 else None)]
            vel = w_ * vel + c1 * rng.random((n, 2)) * (pb - pos) + c2 * rng.random((n, 2)) * (ld - pos)
            pos = np.clip(pos + vel, lb, ub); Fn = F(pos)
            dom = np.all(Fn <= pbF, 1); pb[dom] = pos[dom]; pbF[dom] = Fn[dom]
            A, AF = upd(A, AF, pos, Fn)
        return A, AF

    def igd(Pa):
        return np.sqrt(((norm(Pg)[:, None, :] - norm(Pa)[None, :, :]) ** 2).sum(-1)).min(1).mean()

    w_base = [3, 3, 1, 1]
    nat_g = np.where(MAXI, -Pg, Pg)
    gb = np.argmax(desir(nat_g, w_base))
    runs, fronts = [], {}
    for alg, fn in [('MOPSO', run_mopso), ('NSGA-III', run_nsga3)]:
        for s in range(seeds):
            Xa, Fa = fn(s)
            k = NonDominatedSorting().do(Fa, only_non_dominated_front=True)
            Xa, Fa = Xa[k], Fa[k]
            nat = np.where(MAXI, -Fa, Fa); b = np.argmax(desir(nat, w_base))
            runs.append(dict(alg=alg, seed=s, n=len(Fa), HV_ratio=hv(norm(Fa)) / hv_grid, IGD=igd(Fa),
                             best_rpm=Xa[b, 0], best_blend=Xa[b, 1]))
            if s == 0: fronts[alg] = (Xa, nat)
        print(f"{alg}: {seeds} seeds done")
    R = pd.DataFrame(runs)
    R.to_csv(f'{out_dir}/four_obj_algo_runs.csv', index=False)

    def pm(x, dec):
        return f'{x.mean():.{dec}f} ± {x.std(ddof=1):.{dec}f}'
    t10 = [['Grid search (benchmark)', len(Pg), '1.0000', '0', f'{Xg_front[gb, 0]:.0f}', f'{Xg_front[gb, 1]:.2f}']]
    for alg, lab in [('MOPSO', 'MOPSO'), ('NSGA-III', 'NSGA-III')]:
        g = R[R.alg == alg]
        hd = 5 if g.HV_ratio.std(ddof=1) < 1e-4 else 4
        t10.append([f'{lab} ({seeds} seeds)', f'{g.n.mean():.0f}', pm(g.HV_ratio, hd), pm(g.IGD, 4),
                    pm(g.best_rpm, 0), pm(g.best_blend, 2)])
    t10 = pd.DataFrame(t10, columns=['Method', 'Non-dominated points', 'HV / HV(grid), mean ± SD',
                                     'IGD, mean ± SD', 'Best-desirability speed (rpm)', 'Best-desirability blend (%)'])
    t10.to_csv(f'{out_dir}/table10.csv', index=False, encoding='utf-8-sig')
    print("\nTable 10"); print(t10.to_string(index=False))

    sets = [('3, 3, 1, 1 (baseline)', [3, 3, 1, 1]), ('1, 1, 1, 1 (equal)', [1, 1, 1, 1]),
            ('4, 1, 1, 1 (economy-first)', [4, 1, 1, 1]), ('1, 4, 1, 1 (power-first)', [1, 4, 1, 1]),
            ('1, 1, 3, 3 (emissions-first)', [1, 1, 3, 3])]
    rows11 = []
    for lab, w in sets:
        D = desir(nat_g, w); i = D.argmax()
        rows11.append([lab, f'{Xg_front[i, 0]:.0f}', f'{Xg_front[i, 1]:.2f}', f'{D[i]:.3f}'])
    t11 = pd.DataFrame(rows11, columns=['Weights (BSFC, power, CO, NOₓ)', 'Best speed (rpm)',
                                        'Best blend (%)', 'Composite desirability'])
    t11.to_csv(f'{out_dir}/table11.csv', index=False, encoding='utf-8-sig')
    print("\nTable 11"); print(t11.to_string(index=False))

    best = dict(rpm=Xg_front[gb, 0], blend=Xg_front[gb, 1], bsfc=nat_g[gb, 0], power=nat_g[gb, 1],
                co=nat_g[gb, 2], nox=nat_g[gb, 3])
    tested = []
    for nm, (r_, b_) in TESTED_POINTS.items():
        v = F(np.array([[r_, b_]]))[0]; v = np.where(MAXI, -v, v)
        tested.append(dict(rpm=r_, blend=b_, bsfc=v[0], power=v[1], co=v[2], nox=v[3]))
    tested = pd.DataFrame(tested)

    def frame(Xa, nat):
        return pd.DataFrame(dict(rpm=Xa[:, 0], blend=Xa[:, 1], bsfc=nat[:, 0], power=nat[:, 1], co=nat[:, 2], nox=nat[:, 3]))
    gridf = frame(Xg_front, nat_g)
    mop, nsg = frame(*fronts['MOPSO']), frame(*fronts['NSGA-III'])
    pd.concat([gridf.assign(set='grid'), mop.assign(set='MOPSO'), nsg.assign(set='NSGA-III'),
               tested.assign(set='tested'), pd.DataFrame([best]).assign(set='best')]).to_csv(
               f'{out_dir}/fig15_data.csv', index=False)

    C_GRID, C_MOP, C_NSG, C_BEST, C_TEST = '#B9BEC6', '#0072B2', '#D55E00', '#111111', '#F2B705'
    style = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
             'font.size': 8, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
             'mathtext.default': 'regular', 'axes.linewidth': 0.7, 'axes.edgecolor': '#333333',
             'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
             'grid.color': '#E4E4E4', 'grid.linewidth': 0.5, 'axes.axisbelow': True,
             'savefig.facecolor': 'white', 'figure.dpi': 100, 'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(style):
        fig, axs = plt.subplots(1, 3, figsize=(7.5, 3.0), constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.06, h_pad=0.04, wspace=0.07)

        def draw(ax, x, y):
            ax.scatter(gridf[x], gridf[y], s=6, c=C_GRID, linewidths=0, zorder=1)
            ax.scatter(mop[x], mop[y], s=15, facecolors='none', edgecolors=C_MOP, linewidths=0.8, zorder=2)
            ax.scatter(nsg[x], nsg[y], s=26, c=C_NSG, marker='^', edgecolors='white', linewidths=0.4, zorder=3)
            ax.scatter(tested[x], tested[y], s=80, c=C_TEST, marker='*', edgecolors='#111111', linewidths=0.6, zorder=5)
            ax.scatter(best[x], best[y], s=34, c=C_BEST, marker='D', edgecolors='white', linewidths=0.6, zorder=6)

        arrow = dict(arrowstyle='-', color='#888888', lw=0.6, shrinkA=0, shrinkB=2)
        a = axs[0]; draw(a, 'power', 'bsfc')
        a.set_xlabel('Brake power (W)'); a.set_ylabel('BSFC (kg/kWh)')
        a.annotate('fuel-economy\nend', xy=(gridf.power.min(), gridf.bsfc.min()),
                   xytext=(gridf.power.min() + 60, gridf.bsfc.min() + 0.017), fontsize=7, color='#444444',
                   ha='center', va='center', arrowprops=arrow)
        a.annotate('power\nend', xy=(gridf.power.max(), gridf.bsfc.max()),
                   xytext=(gridf.power.max() - 95, gridf.bsfc.max()), fontsize=7, color='#444444',
                   ha='center', va='center', arrowprops=arrow)

        b = axs[1]; draw(b, 'co', 'nox')
        b.set_xlabel('Brake-specific CO (g/kWh)'); b.set_ylabel(r'Brake-specific NO$_x$ (g/kWh)')

        c = axs[2]; draw(c, 'rpm', 'blend')
        c.axvline(2200, color='#777777', lw=0.7, ls=(0, (4, 3)), zorder=0)
        c.set_xlabel('Engine speed (rpm)'); c.set_ylabel('Biodiesel blend (%)')
        c.annotate(f"{best['rpm']:.0f} rpm, {best['blend']:.1f} %", xy=(best['rpm'], best['blend']),
                   xytext=(2025, c.get_ylim()[1] - 0.4), fontsize=7, color='#222222', ha='left', va='center',
                   arrowprops=dict(arrowstyle='-', color='#666666', lw=0.6, shrinkA=0, shrinkB=3))
        for ax, t in zip(axs, 'abc'):
            ax.text(-0.02, 1.04, t, transform=ax.transAxes, fontsize=10.5, fontweight='bold', ha='right', va='bottom')

        h = [Line2D([], [], marker='o', ls='', ms=3, mfc=C_GRID, mec=C_GRID, label=f'Grid search ({len(gridf)} points)'),
             Line2D([], [], marker='o', ls='', ms=4.5, mfc='none', mec=C_MOP, mew=0.8, label=f'MOPSO ({len(mop)})'),
             Line2D([], [], marker='^', ls='', ms=5, mfc=C_NSG, mec='white', mew=0.4, label=f'NSGA-III ({len(nsg)})'),
             Line2D([], [], marker='*', ls='', ms=9, mfc=C_TEST, mec='#111111', mew=0.6, label='Engine-tested (predicted)'),
             Line2D([], [], marker='D', ls='', ms=5, mfc=C_BEST, mec='white', mew=0.6, label='Highest desirability')]
        fig.legend(handles=h, loc='outside lower center', ncol=5, frameon=False, fontsize=7.5,
                   handletextpad=0.3, columnspacing=1.4)
        for ext, kw in [('png', dict(dpi=600)), ('tif', dict(dpi=300, pil_kwargs={'compression': 'tiff_lzw'})),
                        ('pdf', {}), ('eps', {})]:
            fig.savefig(f'{out_dir}/Fig15.{ext}', **kw)
        plt.close(fig)
    print(f"Saved: Fig15 (png/tif/pdf/eps), table10.csv, table11.csv in {out_dir}")

print("\nRunning four-objective Pareto analysis (Table 10, Table 11, Fig 15)... this takes several minutes.")
try:
    run_four_objective_analysis(df, save_folder)
except ImportError:
    print("pymoo is not installed. Run:  pip install pymoo   then rerun to get Table 10, Table 11 and Fig 15.")
