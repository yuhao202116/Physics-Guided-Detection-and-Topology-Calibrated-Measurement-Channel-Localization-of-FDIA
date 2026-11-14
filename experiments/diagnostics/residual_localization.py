#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-case localization analysis using PGAE-tau (shared learned temperature).
Usage: PGAE_CASE=case14 python3 experiments/diagnostics/residual_localization.py case14 [save_root]
"""
import sys, os
import argparse

parser = argparse.ArgumentParser(description="PGAE-tau residual localization analysis.")
parser.add_argument("case", choices=("case14", "case30", "case57"))
parser.add_argument("save_root", nargs="?", default=None)
args = parser.parse_args()
CASE_NAME = args.case

os.environ['PGAE_CASE'] = CASE_NAME
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
import warnings, json
warnings.filterwarnings('ignore')

from data.loaders import load_case, load_dataset
from configs.config_mea_idx import define_mea_idx_noise
from physics.attacks import FDI
from data.dataset import scaler
from configs.nn_setting import nn_setting
from models.detection.pgae_tau import PGAE_tau
from configs.paths import OUTPUT_ROOT, checkpoint_path

SAVE_ROOT = args.save_root or str(OUTPUT_ROOT / "pgae_localization")

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

MODEL_LABEL = r'PGAE-$\tau$'


def build_mea_labels(case_class):
    type_order = ['pf', 'pt', 'pi', 'vang', 'qf', 'qt', 'qi', 'vmag']
    labels = []
    for mt in type_order:
        n = len(case_class.idx.get(mt, []))
        for k in range(n):
            if mt == 'pi': labels.append(f'P_inj_Bus{k+1}')
            elif mt == 'pf':
                fbus = int(case_class.case['branch'][k, 0])
                tbus = int(case_class.case['branch'][k, 1])
                labels.append(f'P_flow_{fbus}→{tbus}')
            elif mt == 'qi': labels.append(f'Q_inj_Bus{k+1}')
            elif mt == 'qf':
                fbus = int(case_class.case['branch'][k, 0])
                tbus = int(case_class.case['branch'][k, 1])
                labels.append(f'Q_flow_{fbus}→{tbus}')
            else: labels.append(f'{mt}_{k}')
    return labels


def single_sample_analysis(model, attacker, scaler_inst, device,
                           test_loader, mea_labels, save_dir):
    fig_dir = os.path.join(save_dir, 'single_sample')
    os.makedirs(fig_dir, exist_ok=True)

    for _, (idx, input_data, v_est_pre, v_est_last) in enumerate(test_loader):
        z_clean = input_data[0, -1].numpy(); v_est = v_est_last.flatten().numpy(); break

    z_att_col, _ = attacker.gen_fdi_att(z_noise=np.expand_dims(z_clean, axis=1), v_est=v_est,
                                         ang_no=2, mag_no=2, ang_str=0.3, mag_str=0.3)
    z_att = z_att_col.flatten(); true_pert = np.abs(z_att - z_clean)

    zb = input_data.clone(); zb[0, -1] = torch.from_numpy(z_att).float()
    zs = torch.from_numpy((zb.numpy() - scaler_inst.min) / (scaler_inst.max - scaler_inst.min + 1e-8)).float().to(device)
    with torch.no_grad(): _, vr, vi = model(zs)
    vp = (vr[0] + 1j * vi[0]).cpu().numpy()
    pg_res = np.abs(z_att - np.real(attacker.h_x_pypower(vp)).flatten())

    N_feat = len(pg_res)
    corr = np.corrcoef(true_pert, pg_res)[0, 1]
    pg_rank = np.argsort(np.argsort(-pg_res)); true_rank = np.argsort(np.argsort(-true_pert))
    r_corr = np.corrcoef(true_rank, pg_rank)[0, 1]

    # Fig 1: Bars
    K = min(20, N_feat); pg_top = np.argsort(-pg_res)[:K]; true_top = np.argsort(-true_pert)[:K]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5)); x = np.arange(K)
    ax = axes[0]
    ax.bar(x - 0.2, pg_res[pg_top], 0.4, label='PGAE Residual', color='#6C5CE7', alpha=0.85)
    ax.bar(x + 0.2, true_pert[pg_top], 0.4, label='True Perturbation', color='#00B894', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([mea_labels[i][:22] for i in pg_top], rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Magnitude'); ax.set_title(f'Top-{K} by PGAE', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
    ax = axes[1]
    ax.bar(x - 0.2, true_pert[true_top], 0.4, label='True Perturbation', color='#00B894', alpha=0.85)
    ax.bar(x + 0.2, pg_res[true_top], 0.4, label='PGAE Residual', color='#6C5CE7', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([mea_labels[i][:22] for i in true_top], rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Magnitude'); ax.set_title(f'Top-{K} by True', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
    fig.suptitle(f'{CASE_NAME} | {MODEL_LABEL} | FDI 30% | Pearson r={corr:.3f}', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'residual_bars.pdf')); plt.close()

    # Fig 2: Scatter + Rank
    pg15 = set(np.argsort(-pg_res)[:15]); true15 = set(np.argsort(-true_pert)[:15])
    ovr15 = len(pg15 & true15) / 15
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(true_pert, pg_res, alpha=0.4, s=12, c='#6C5CE7', edgecolors='none')
    ax.set_xlabel('True Perturbation'); ax.set_ylabel('PGAE Residual')
    ax.set_title('Residual vs Ground Truth', fontweight='bold'); ax.grid(alpha=0.3)
    ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax = axes[1]; hl = pg15 | true15
    colors_hl = ['#00B894' if i in hl else '#DFE6E9' for i in range(N_feat)]
    ax.scatter(true_rank, pg_rank, c=colors_hl, alpha=0.6, s=15, edgecolors='none')
    ax.plot([0, N_feat], [0, N_feat], '--', color='gray', alpha=0.5)
    ax.set_xlabel('True Rank'); ax.set_ylabel(f'{MODEL_LABEL} Rank')
    ax.set_title('Rank Correlation (0=largest)', fontweight='bold'); ax.grid(alpha=0.3)
    ax.text(0.05, 0.95, f'Spearman ρ = {r_corr:.3f}\nTop-15 Ovr = {ovr15:.2%}',
            transform=ax.transAxes, fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.suptitle(f'{CASE_NAME} | {MODEL_LABEL} | FDI 30%', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'scatter_rank.pdf')); plt.close()

    # Fig 3: Heatmap
    type_order = ['pf','pt','pi','vang','qf','qt','qi','vmag']
    # Publication-style notation: P_ij/Q_ij denote directed branch flow
    # from bus i to j; P_i/Q_i with the inj superscript denote bus injection.
    type_names = [
        r'$P_{ij}$', r'$P_{ji}$', r'$P_{i}^{\mathrm{inj}}$', r'$\theta_i$',
        r'$Q_{ij}$', r'$Q_{ji}$', r'$Q_{i}^{\mathrm{inj}}$', r'$|V_i|$',
    ]
    type_data = {}; curr = 0
    for mt, tn in zip(type_order, type_names):
        n = len(attacker.idx.get(mt, []))
        if n > 0: type_data[tn] = {'true': true_pert[curr:curr+n], 'pg': pg_res[curr:curr+n]}
        curr += n
    fig, axes = plt.subplots(2, 1, figsize=(14, max(5, len(type_data)*0.7)))
    all_types = list(type_data.keys()); ml_len = max(len(v['true']) for v in type_data.values())
    for row, (dk, title) in enumerate([('true','True Perturbation'),('pg','PGAE Residual')]):
        ax = axes[row]; hm = np.full((len(all_types), ml_len), np.nan)
        for i, tn in enumerate(all_types): vals = type_data[tn][dk]; hm[i,:len(vals)] = vals
        vmax = max(np.nanmax(hm), 1e-3)
        im = ax.imshow(hm, aspect='auto', cmap='YlOrRd', norm=Normalize(vmin=0, vmax=vmax))
        ax.set_yticks(np.arange(len(all_types))); ax.set_yticklabels(all_types)
        tick_step = max(1, int(np.round(ml_len / 8)))
        tick_positions = np.arange(tick_step - 1, ml_len, tick_step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions + 1)
        ax.set_xlabel('Measurement position within category'); ax.set_title(title, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'{CASE_NAME} | {MODEL_LABEL} | Residual by Type', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'type_heatmap.pdf')); plt.close()

    # Fig 4: Bus-level
    n_bus = attacker.no_bus; bus_r, bus_t = np.zeros(n_bus), np.zeros(n_bus); curr = 0
    for mt in type_order:
        n = len(attacker.idx.get(mt, []))
        for k in range(n):
            if mt in ('pi','qi','vmag','vang'): bi = k
            elif mt in ('pf','qf'): bi = int(attacker.case['branch'][k,0]) - 1
            elif mt in ('pt','qt'): bi = int(attacker.case['branch'][k,1]) - 1
            else: curr += 1; continue
            if bi < n_bus: bus_r[bi] += pg_res[curr]; bus_t[bi] += true_pert[curr]
            curr += 1
    fig, ax = plt.subplots(figsize=(max(8, n_bus*0.35), 5))
    x = np.arange(n_bus); w = 0.35
    ax.bar(x - w/2, bus_t, w, label='True Perturbation', color='#00B894', alpha=0.8)
    ax.bar(x + w/2, bus_r, w, label='PGAE Residual', color='#6C5CE7', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f'Bus {i+1}' for i in range(n_bus)], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Aggregated Magnitude'); ax.set_title(f'{MODEL_LABEL}: Bus-Level', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'bus_level.pdf')); plt.close()

    print(f'    [✓] 4 figures (bars, scatter, heatmap, bus-level)')
    return {'pearson_r': float(corr), 'spearman_rho': float(r_corr), 'top15_overlap': float(ovr15)}


def multi_sample_evaluation(model, attacker, scaler_inst, device,
                            test_loader, save_dir, N_samples=300, K_values=[5,10,15,20,30]):
    fig_dir = os.path.join(save_dir, 'statistics'); os.makedirs(fig_dir, exist_ok=True)
    metrics = {k: {'overlap': [], 'precision': [], 'recall': []} for k in K_values}
    spearman_all, pearson_all = [], []

    print(f'    Evaluating {N_samples} samples...')
    for si, (idx, input_data, v_est_pre, v_est_last) in tqdm(enumerate(test_loader), total=N_samples):
        if si >= N_samples: break
        zc = input_data[0, -1].numpy(); ve = v_est_last.flatten().numpy()
        ac, _ = attacker.gen_fdi_att(z_noise=np.expand_dims(zc, axis=1), v_est=ve, ang_no=2, mag_no=2, ang_str=0.3, mag_str=0.3)
        za = ac.flatten(); tp = np.abs(za - zc)
        zb = input_data.clone(); zb[0, -1] = torch.from_numpy(za).float()
        zs = torch.from_numpy((zb.numpy() - scaler_inst.min) / (scaler_inst.max - scaler_inst.min + 1e-8)).float().to(device)
        with torch.no_grad(): _, vr, vi = model(zs)
        vp = (vr[0] + 1j * vi[0]).cpu().numpy()
        pr = np.abs(za - np.real(attacker.h_x_pypower(vp)).flatten())
        prk = np.argsort(np.argsort(-pr)); trk = np.argsort(np.argsort(-tp))
        spearman_all.append(np.corrcoef(trk, prk)[0, 1]); pearson_all.append(np.corrcoef(tp, pr)[0, 1])
        ta = set(np.where(tp > 1e-9)[0]); nt = len(ta)
        for k in K_values:
            pt = set(np.argsort(-pr)[:k])
            metrics[k]['overlap'].append(len(pt & ta) / k)
            metrics[k]['precision'].append(len(pt & ta) / k)
            metrics[k]['recall'].append(len(pt & ta) / nt if nt > 0 else 0)

    results = {}
    for k in K_values:
        for mk in ['overlap','precision','recall']:
            arr = np.array(metrics[k][mk])
            results[f'{mk}_{k}'] = {'mean': float(arr.mean()), 'std': float(arr.std()), 'median': float(np.median(arr))}
    results['spearman'] = {'mean': float(np.mean(spearman_all)), 'std': float(np.std(spearman_all))}
    results['pearson'] = {'mean': float(np.mean(pearson_all)), 'std': float(np.std(pearson_all))}

    print(f'    === {MODEL_LABEL} (N={N_samples}) ===')
    for k in K_values:
        print(f'      Top-{k}: Ovr={results[f"overlap_{k}"]["mean"]:.3f}±{results[f"overlap_{k}"]["std"]:.3f}  '
              f'Prec={results[f"precision_{k}"]["mean"]:.3f}  Rec={results[f"recall_{k}"]["mean"]:.3f}')
    print(f'      Spearman ρ={results["spearman"]["mean"]:.3f}±{results["spearman"]["std"]:.3f}  '
          f'Pearson r={results["pearson"]["mean"]:.3f}±{results["pearson"]["std"]:.3f}')

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    colors = ['#6C5CE7','#00B894','#FDCB6E','#E17055','#00CEC9']
    for i, k in enumerate(K_values[:3]):
        ax = axes[0, i]; arr = np.array(metrics[k]['overlap'])
        ax.hist(arr, bins=20, color=colors[i], alpha=0.7, edgecolor='white')
        ax.axvline(arr.mean(), color='red', ls='--', lw=1.5)
        ax.set_xlabel(f'Top-{k} Overlap'); ax.set_ylabel('Count')
        ax.set_title(f'Top-{k}: {arr.mean():.3f}±{arr.std():.3f}', fontweight='bold'); ax.grid(alpha=0.3)
    ax = axes[1,0]
    ax.hist(spearman_all, bins=20, color='#00B894', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(spearman_all), color='red', ls='--', lw=1.5)
    ax.set_xlabel('Spearman ρ'); ax.set_ylabel('Count')
    ax.set_title(f'Spearman: {np.mean(spearman_all):.3f}±{np.std(spearman_all):.3f}', fontweight='bold'); ax.grid(alpha=0.3)
    ax = axes[1,1]
    ax.hist(pearson_all, bins=20, color='#6C5CE7', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(pearson_all), color='red', ls='--', lw=1.5)
    ax.set_xlabel('Pearson r'); ax.set_ylabel('Count')
    ax.set_title(f'Pearson: {np.mean(pearson_all):.3f}±{np.std(pearson_all):.3f}', fontweight='bold'); ax.grid(alpha=0.3)
    ax = axes[1,2]
    ax.scatter(metrics[15]['precision'], metrics[15]['recall'], alpha=0.4, s=10, c='#6C5CE7')
    ax.set_xlabel('Precision@15'); ax.set_ylabel('Recall@15')
    ax.set_title('Precision vs Recall (K=15)', fontweight='bold'); ax.grid(alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.suptitle(f'{CASE_NAME} | {MODEL_LABEL} | Multi-Sample (N={N_samples})', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'multisample_stats.pdf')); plt.close()
    print('    [✓] Multi-sample figure')
    return results


def strength_sweep(model, attacker, scaler_inst, device, test_loader, save_dir, N_samples=200):
    sd = os.path.join(save_dir, 'strength_sweep'); os.makedirs(sd, exist_ok=True)
    strengths = [0.1, 0.2, 0.3, 0.4, 0.5]; sw = {}
    for s in strengths:
        ov, sp = [], []
        for si, (idx, inp, vp, vl) in enumerate(test_loader):
            if si >= N_samples: break
            zc = inp[0,-1].numpy(); ve = vl.flatten().numpy()
            ac, _ = attacker.gen_fdi_att(z_noise=np.expand_dims(zc, axis=1), v_est=ve, ang_no=2, mag_no=2, ang_str=s, mag_str=s)
            za = ac.flatten(); tp = np.abs(za - zc)
            zb = inp.clone(); zb[0,-1] = torch.from_numpy(za).float()
            zs = torch.from_numpy((zb.numpy() - scaler_inst.min) / (scaler_inst.max - scaler_inst.min + 1e-8)).float().to(device)
            with torch.no_grad(): _, vr, vi = model(zs)
            vp = (vr[0] + 1j * vi[0]).cpu().numpy()
            pr = np.abs(za - np.real(attacker.h_x_pypower(vp)).flatten())
            prk = np.argsort(np.argsort(-pr)); trk = np.argsort(np.argsort(-tp))
            ov.append(len(set(np.argsort(-pr)[:15]) & set(np.where(tp > 1e-9)[0])) / 15)
            sp.append(np.corrcoef(trk, prk)[0, 1])
        sw[s] = {'overlap_mean': float(np.mean(ov)), 'overlap_std': float(np.std(ov)),
                 'spearman_mean': float(np.mean(sp)), 'spearman_std': float(np.std(sp))}
        print(f'    {s*100:.0f}%: Ovr={sw[s]["overlap_mean"]:.3f} Sp={sw[s]["spearman_mean"]:.3f}')
    fig, ax = plt.subplots(figsize=(8,5))
    x = [s*100 for s in strengths]
    ax.errorbar(x, [sw[s]['overlap_mean'] for s in strengths], [sw[s]['overlap_std'] for s in strengths],
                marker='o', label='Top-15 Overlap', color='#6C5CE7', capsize=5, lw=2)
    ax.errorbar(x, [sw[s]['spearman_mean'] for s in strengths], [sw[s]['spearman_std'] for s in strengths],
                marker='s', label='Spearman ρ', color='#00B894', capsize=5, lw=2)
    ax.set_xlabel('Attack Strength (%)'); ax.set_ylabel('Metric Value')
    ax.set_title(f'{CASE_NAME} | {MODEL_LABEL} | Strength Sensitivity', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(sd, 'strength_sweep.pdf')); plt.close()
    print('    [✓] Strength sweep')
    return sw


if __name__ == '__main__':
    print(f'\n{"="*60}\n  Case: {CASE_NAME}\n{"="*60}')

    case_class = load_case()
    _, test_loader, _, _ = load_dataset()
    from configs.config import sys_config
    idx_, _, ns_ = define_mea_idx_noise(case_class.case, sys_config['measure_type'])
    attacker = FDI(case_class.case, ns_, idx_, sys_config['fpr'])
    si = scaler(); device = nn_setting['device']
    labels = build_mea_labels(case_class)
    print(f'  Feature dim: {len(labels)} | Buses: {case_class.no_bus} | Device: {device}')

    save_dir = os.path.join(SAVE_ROOT, CASE_NAME, 'localization')
    os.makedirs(save_dir, exist_ok=True)

    ckpt = checkpoint_path("PGAE_tau.pt")
    if not os.path.exists(ckpt):
        print(f'No checkpoint at {ckpt}. Please train PGAE-tau first.')
        sys.exit(1)

    print(f'\n  --- {MODEL_LABEL} ---')
    model = PGAE_tau().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    print(f'  Single-sample analysis...')
    ss = single_sample_analysis(model, attacker, si, device, test_loader, labels, save_dir)
    print(f'  Multi-sample evaluation...')
    np.random.seed(42)
    ms = multi_sample_evaluation(model, attacker, si, device, test_loader, save_dir, N_samples=300)
    print(f'  Attack strength sweep...')
    np.random.seed(42)
    sw = strength_sweep(model, attacker, si, device, test_loader, save_dir, N_samples=200)

    all_metrics = {'PGAE-tau': {'single_sample': ss, 'multisample': ms, 'strength_sweep': sw}}

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.integer,)): return int(obj)
            return super().default(obj)

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2, cls=NpEncoder)

    print(f'\n  ✅ {CASE_NAME} complete.')
