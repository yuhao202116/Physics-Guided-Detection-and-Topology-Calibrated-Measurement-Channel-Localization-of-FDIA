#!/usr/bin/env python3
"""
Draw a schematic of sparse physical-decoder dependencies and residual tracing.
The diagram uses an illustrative topology to show Jacobian structure.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.size': 9, 'font.family': 'sans-serif',
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300,
})

fig = plt.figure(figsize=(17, 7.8))

# =====================================================================
# LEFT: System topology schematic (Bus 5 & 10 attack -> affected region)
# =====================================================================
ax1 = fig.add_axes([0.02, 0.06, 0.40, 0.90])
ax1.set_xlim(-2, 12); ax1.set_ylim(-2, 12)
ax1.axis('off')

bus_pos = {
    1: (0.5, 10),  2: (4, 10.5),  3: (7, 10),
    4: (1, 8),     5: (4, 8.5),   6: (7, 8),    7: (2, 5.5),
    8: (5, 6),     9: (8, 6),    10: (3, 3),
    11: (6, 3.5), 12: (9, 3),    13: (11, 6),   14: (11, 8.5),
}

branches = [
    (1,2), (1,4), (2,3), (2,5), (3,6), (4,5),
    (4,7), (5,6), (5,8), (6,9), (7,8), (7,10),
    (8,9), (8,11), (9,12), (10,11), (11,13), (12,14)
]

attacked = {5, 10}
neighbors = set()
for b in attacked:
    for (i, j) in branches:
        if i == b: neighbors.add(j)
        if j == b: neighbors.add(i)
affected = attacked | neighbors

# Draw branches
for (i, j) in branches:
    xi, yi = bus_pos[i]; xj, yj = bus_pos[j]
    in_affected = (i in affected) or (j in affected)
    color = '#E17055' if in_affected else '#DFE6E9'
    lw = 2.2 if in_affected else 1.0
    ax1.plot([xi, xj], [yi, yj], color=color, lw=lw, zorder=1, alpha=0.7)

# Draw buses
for b, (x, y) in bus_pos.items():
    if b in attacked:
        fc, ec, sz, lw = '#D63031', '#B33939', 280, 2.5
    elif b in neighbors:
        fc, ec, sz, lw = '#FDCB6E', '#E0A800', 220, 2
    else:
        fc, ec, sz, lw = '#DFE6E9', '#B2BEC3', 180, 1
    ax1.scatter(x, y, s=sz, c=fc, edgecolors=ec, linewidth=lw, zorder=3)
    ax1.text(x, y-0.5, str(b), fontsize=7, ha='center', va='top',
             color='#2D3436', fontweight='bold')

# Legend
ax1.scatter(-10, -10, s=100, c='#D63031', edgecolors='#B33939',
            linewidth=1.5, label='Attacked Bus')
ax1.scatter(-10, -10, s=90, c='#FDCB6E', edgecolors='#E0A800',
            linewidth=1.5, label='1-Hop Neighbor')
ax1.scatter(-10, -10, s=70, c='#DFE6E9', edgecolors='#B2BEC3',
            linewidth=1, label='Unaffected')
ax1.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=1)

ax1.annotate('Bus 5', xy=bus_pos[5], xytext=(bus_pos[5][0]-1.8, bus_pos[5][1]+1),
             fontsize=8, color='#D63031', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#D63031', lw=1))
ax1.annotate('Bus 10', xy=bus_pos[10], xytext=(bus_pos[10][0]+1.8, bus_pos[10][1]-1),
             fontsize=8, color='#D63031', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#D63031', lw=1))

ax1.set_title('(a) Attack Impact: Limited to Electrical Neighborhood',
              fontsize=10, fontweight='bold', pad=8)

# Red dashed circle around affected region
cx, cy = 5, 5.5
circle = plt.Circle((cx, cy), 4.5, fill=False, edgecolor='#E17055',
                     linestyle='--', linewidth=1.5, alpha=0.5)
ax1.add_patch(circle)
ax1.text(cx+2.2, cy+3.8, 'Affected\nRegion', fontsize=7,
         color='#E17055', ha='center', alpha=0.8)


# =====================================================================
# RIGHT: Jacobian sparsity matrix + measurement mapping
# =====================================================================
ax2 = fig.add_axes([0.47, 0.06, 0.52, 0.90])
ax2.set_xlim(-1, 23); ax2.set_ylim(-1, 16.5)
ax2.axis('off')

# ---- Jacobian matrix ----
mat_x, mat_y = -0.5, 13.5
mat_w, mat_h = 6.5, -10.5

rect = mpatches.FancyBboxPatch((mat_x, mat_y), mat_w, mat_h,
                                boxstyle="round,pad=0.1", edgecolor='#2D3436',
                                facecolor='#F8F9FA', linewidth=1.2)
ax2.add_patch(rect)

# Matrix title
ax2.text(mat_x + mat_w/2, mat_y + 0.55,
         r'$\mathbf{J} = \frac{\partial \mathbf{h}}{\partial \mathbf{V}}$',
         fontsize=11, ha='center', fontweight='bold', color='#2D3436')
# Axis labels
ax2.text(mat_x - 0.7, mat_y + mat_h/2, 'Meas.\n$F$', fontsize=7,
         ha='center', va='center', rotation=90, color='#636E72')
ax2.text(mat_x + mat_w/2, mat_y + mat_h - 0.35, 'Buses  $N$', fontsize=7,
         ha='center', color='#636E72')

# Bus 5 column block (wider red block - attacked)
bx5, by5, bw, bh5 = mat_x + 1.4, mat_y - 1.6, 1.1, -2.3
patch5 = mpatches.FancyBboxPatch((bx5, by5), bw, bh5,
    boxstyle="round,pad=0.05", edgecolor='#D63031',
    facecolor='#FF7675', alpha=0.55, linewidth=1.5)
ax2.add_patch(patch5)
ax2.text(bx5 + bw/2, by5 + bh5 - 0.45,
         r'$\mathbf{V}_{\!5}$', fontsize=8, ha='center',
         color='#D63031', fontweight='bold')

# Bus 10 column block
bx10, by10, bw, bh10 = mat_x + 4.15, mat_y - 1.6, 1.1, -2.3
patch10 = mpatches.FancyBboxPatch((bx10, by10), bw, bh10,
    boxstyle="round,pad=0.05", edgecolor='#D63031',
    facecolor='#FF7675', alpha=0.55, linewidth=1.5)
ax2.add_patch(patch10)
ax2.text(bx10 + bw/2, by10 + bh10 - 0.45,
         r'$\mathbf{V}_{\!10}$', fontsize=8, ha='center',
         color='#D63031', fontweight='bold')

# Thin gray columns for non-attacked buses
gray_blocks = np.linspace(mat_x + 0.3, mat_x + mat_w - 0.7, 9)
for gx in gray_blocks:
    if abs(gx - bx5) < 1.3 or abs(gx - bx10) < 1.3:
        continue
    p = mpatches.FancyBboxPatch((gx, mat_y - 0.1), 0.45, -0.9,
        boxstyle="round,pad=0.02", edgecolor='#B2BEC3',
        facecolor='#DFE6E9', alpha=0.4, linewidth=0.6)
    ax2.add_patch(p)

# Ellipsis
ax2.text(mat_x + 0.9, mat_y - 0.5, '⋯', fontsize=11, ha='center', color='#636E72')
ax2.text(mat_x + 3.3, mat_y - 0.55, '⋯', fontsize=11, ha='center', color='#636E72')
ax2.text(mat_x + 5.5, mat_y - 0.55, '⋯', fontsize=11, ha='center', color='#636E72')

# Row structure indicator (thin horizontal bars left of matrix)
row_ys = np.linspace(mat_y - 1.0, mat_y + mat_h + 0.8, 8)
for ry in row_ys[:6]:
    ax2.plot([mat_x - 1.2, mat_x], [ry, ry], color='#B2BEC3', lw=0.8, alpha=0.5)
    ax2.text(mat_x - 1.5, ry, r'$\mathbf{r}_k$', fontsize=5.5, ha='center',
             va='center', color='#636E72', alpha=0.7)

# ---- Measurement vector (right side) ----
x_mea = 9.5
mea_y0 = 14.0
mea_width = 3.8   # Leave room for residual markers.
mea_items = [
    ('P_flow_$_{\\rm from}$',  14, '#74B9FF'),
    ('P_flow_$_{\\rm to}$',    14, '#74B9FF'),
    ('P_inj',                  14, '#0984E3'),
    ('Q_flow_$_{\\rm from}$',  14, '#81ECEC'),
    ('Q_flow_$_{\\rm to}$',    14, '#81ECEC'),
    ('Q_inj',                  14, '#00CEC9'),
]
mea_gap = 0.95
for i, (label, count, color) in enumerate(mea_items):
    y = mea_y0 - i * mea_gap
    rm = mpatches.FancyBboxPatch((x_mea, y - 0.3), mea_width, 0.6,
        boxstyle="round,pad=0.06", edgecolor=color,
        facecolor=color, alpha=0.18, linewidth=0.8)
    ax2.add_patch(rm)
    ax2.text(x_mea + 0.25, y, label, fontsize=7.5, ha='left', va='center',
             color='#2D3436', fontweight='bold')

# Red star markers — placed to the RIGHT of measurement boxes with clear gap
star_x = x_mea + mea_width + 1.2   # clear gap from measurement boxes
attack_rows = [mea_y0 - 2*mea_gap,   # P_inj
               mea_y0 - 5*mea_gap,   # Q_inj
               mea_y0 - 0*mea_gap,   # P_flow_from
               mea_y0 - 3*mea_gap]   # Q_flow_from
for yy in attack_rows:
    ax2.plot(star_x, yy, marker='*', markersize=15, color='#D63031',
             zorder=5, markeredgewidth=0.5, clip_on=False)

# Star legend — placed above measurement items, away from content
ax2.text(x_mea, mea_y0 + 0.65,
         '★ = Attacked measurement type',
         fontsize=7.5, color='#D63031', fontweight='bold', va='bottom')

# ---- Clean horizontal arrow from Jacobian to measurements ----
arrow_y = 12.5
arrow_x0 = mat_x + mat_w + 0.2   # right edge of Jacobian matrix
arrow_x1 = x_mea - 0.2            # left edge of measurement block
ax2.annotate('', xy=(arrow_x1, arrow_y),
             xytext=(arrow_x0, arrow_y),
             arrowprops=dict(arrowstyle='->', color='#2D3436', lw=1.5))
ax2.text((arrow_x0 + arrow_x1)/2, arrow_y + 0.35,
         r'Residual  $\mathbf{r} = |\mathbf{z} - \mathbf{h}(\mathbf{V})|$',
         fontsize=7.5, ha='center', color='#636E72', fontstyle='italic')

# ---- Residual equation (moved up to avoid crowding) ----
eq_x = x_mea - 2.0
eq_y = 4.8
ax2.text(eq_x, eq_y,
         r'$r_i^{\,\rm (PGAE)} = |\,z_i^{\rm att} - h_i(\mathbf{V}^{\rm PGAE})\,|$',
         fontsize=10, color='#2D3436', fontweight='bold')
ax2.text(eq_x, eq_y - 0.85,
         r'$\approx |\delta_i|$  (attacked $i$)\qquad\qquad'
         r'$\approx |\varepsilon_i|$  (unattacked $i$)',
         fontsize=9, color='#636E72')

# ---- Key insight box (more vertical space below residual equation) ----
box_x, box_y = eq_x - 0.5, 1.0
box_w, box_h = 14.5, 1.6
ibox = mpatches.FancyBboxPatch((box_x, box_y), box_w, box_h,
    boxstyle="round,pad=0.15", edgecolor='#0984E3',
    facecolor='#DFE6E9', alpha=0.4, linewidth=1.2)
ax2.add_patch(ibox)
ax2.text(box_x + box_w/2, box_y + box_h - 0.5,
         r'Sparse $\partial\mathbf{h}/\partial\mathbf{V}$'
         r' $\rightarrow$ Each residual $r_i$'
         r' depends only on its electrically connected $V_j$',
         fontsize=8, ha='center', color='#636E72')
ax2.text(box_x + box_w/2, box_y + 0.4,
         r'$\mathbf{\rightarrow}$ '
         r'\textbf{Localization:} $r_i$ identifies the attack on bus $j$'
         r' via non-zero $\partial h_i/\partial V_j$',
         fontsize=8, ha='center', fontweight='bold', color='#0984E3')

ax2.set_title(r'(b) Sparse Decoder Jacobian $\rightarrow$ Traceable Residuals',
              fontsize=10, fontweight='bold', pad=6)

# =====================================================================
# Global title
# =====================================================================
fig.suptitle('Physical Decoder: Sparse Jacobian Enables Per-Measurement Attack Traceability',
             fontsize=12, fontweight='bold', y=0.998, color='#1F3864')

from pathlib import Path
from configs.paths import OUTPUT_ROOT
figure_dir = OUTPUT_ROOT / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
out_pdf = figure_dir / 'method_jacobian_sparsity.pdf'
out_png = figure_dir / 'method_jacobian_sparsity.png'
plt.savefig(out_pdf, bbox_inches='tight')
print(f'Saved to {out_pdf}')
plt.savefig(out_png, bbox_inches='tight', dpi=150)
print(f'PNG preview: {out_png}')
plt.close()
