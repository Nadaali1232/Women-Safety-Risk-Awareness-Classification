"""Draws reports/figures/project_diagram.png. Numbers match notebooks/Women_Safety_Risk_Awareness.ipynb."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTE = '#1f2933', '#52606d'
YEL, YEL_E = '#FFF0B8', '#1f2933'
BLU, BLU_E = '#CFE4FB', '#2B6CB0'
ORG, ORG_E = '#FFD9BD', '#C05621'
GRN, GRN_E = '#D5F0D2', '#2F855A'
GRY = '#F1F3F5'

fig = plt.figure(figsize=(16, 9)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 90); ax.axis('off')

def box(x, y, w, h, fc='white', ec=INK, lw=1.6, r=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw))
def t(x, y, s, size=10, color=INK, weight='normal', ha='left', va='center'):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va)
def arrow(a, b, color=INK, ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=16, lw=1.7, color=color, ls=ls, shrinkA=0, shrinkB=0))

# frame + title
box(1.5, 1.5, 157, 87, ec=INK, lw=1.8, r=3)
t(6, 82, 'Women Safety Risk-Awareness Classification', 25, INK, 'bold')
t(6, 77.6, 'Machine learning with city-grouped validation and explainability on a 3-class proxy label (low / medium / high)', 12, MUTE)
t(154, 82.5, 'Safety risk-awareness challenge', 11.5, MUTE, ha='right')
t(154, 79.2, '489,804 scenarios · 34 cities', 11.5, MUTE, ha='right')
ax.plot([6, 154], [74.6, 74.6], color='#cbd2d9', lw=1)

# step row
steps = [
 ('1', 'Scenario data',      ['489,804 rows · 40 columns', 'city × day × hour × area × complaint'], YEL, INK),
 ('2', 'Cleaning & EDA',     ['no missing values or duplicates', 'high class only 5% (imbalanced)'], YEL, INK),
 ('3', 'City-grouped CV',    ['5 folds, every test city unseen', '25 of 36 inputs are constant per city'], YEL, INK),
 ('4', 'Modeling & features',['dummy · logistic reg. · LightGBM', 'feature reduction: 36 → 19'], BLU, BLU_E),
 ('5', 'Explainability',     ['SHAP · concept importance', 'surrogate decision tree'], ORG, ORG_E),
]
W, H, GAP, Y0 = 26.5, 11.2, 3.875, 60.5
xs = [6 + i*(W+GAP) for i in range(5)]
for (n, title, lines, fc, ec), x in zip(steps, xs):
    box(x, Y0, W, H, fc, ec)
    ax.add_patch(plt.Circle((x+3, Y0+H-3.1), 1.45, color=INK)); t(x+3, Y0+H-3.1, n, 11, 'white', 'bold', 'center')
    t(x+5.4, Y0+H-3.1, title, 14, INK if ec == INK else ec, 'bold')
    t(x+2, Y0+4.6, lines[0], 9.6, MUTE); t(x+2, Y0+2.1, lines[1], 9.6, MUTE)
    if x != xs[-1]: arrow((x+W+0.3, Y0+H/2), (x+W+GAP-0.3, Y0+H/2))
# dashed connectors to panels
x4 = xs[3]+W/2; x5 = xs[4]+W/2
ax.plot([x4, x4, 31], [Y0, 57.2, 57.2], color=BLU_E, ls='--', lw=1.6); arrow((31, 57.2), (31, 54.3), BLU_E)
ax.plot([x5, x5], [Y0, 57.2], color=ORG_E, ls='--', lw=1.6); arrow((x5, 57.2), (x5, 54.3), ORG_E)

# left panel: results table
PX, PY, PW, PH = 6, 24.5, 73.5, 29.5
box(PX, PY, PW, PH, 'white', BLU_E, 1.8, 2)
t(PX+2, PY+PH-2.6, 'Model comparison (5-fold, unseen cities)', 13.5, BLU_E, 'bold')
chips = [('Log. regression', '36 features', YEL, INK, 1.5),
         ('LightGBM', '36 features', YEL, INK, 1.5),
         ('LightGBM ★ final', '19 selected features', BLU, BLU_E, 2.0)]
cw = 17.4; CC = [PX+27.5, PX+45.5, PX+63.5]
for (a_, b_, fc, ec, lw), c in zip(chips, CC):
    box(c-cw/2, PY+PH-12.2, cw, 6.4, fc, ec, lw, 1.2)
    t(c, PY+PH-8.1, a_, 9.6, INK, 'bold', 'center'); t(c, PY+PH-10.7, b_, 7.8, MUTE, ha='center')
rows = [('Macro-F1',        ['0.697', '0.841', '0.889'], 2),
        ('Accuracy',        ['0.755', '0.896', '0.923'], 2),
        ('High-class F1',   ['0.549', '0.720', '0.815'], 2),
        ('High recall',     ['0.887', '0.639', '0.738'], 0),
        ('High precision',  ['0.397', '0.824', '0.910'], 2),
        ('Weighted kappa',  ['0.662', '0.831', '0.875'], 2)]
ty = PY+PH-14.6; rh = 1.8
box(PX+1.5, PY+1.7, PW-3, ty-PY-1.0, GRY, '#cbd2d9', 1, 1)
ax.add_patch(plt.Rectangle((CC[2]-cw/2-0.4, PY+1.8), cw+0.6, ty-PY-1.2, color=BLU, alpha=0.75, lw=0))
t(PX+3, ty-0.4, 'Metric', 10, INK, 'bold')
for c, h in zip(CC, ['Log. reg.', 'LightGBM 36', 'LightGBM 19']):
    t(c, ty-0.4, h, 10, INK, 'bold', 'center')
for i, (m, vals, best) in enumerate(rows):
    yy = ty-2.9-i*rh
    t(PX+3, yy, m, 10)
    for j, v in enumerate(vals):
        t(CC[j], yy, v, 10.5, INK, 'bold' if j == best else 'normal', 'center')
t(PX+PW/2, PY+0.85, 'Best value per row in bold · the final model has the best score on 5 of 6 rows', 8.6, MUTE, ha='center')

# right panel: explainability
RX, RW = 83.5, 70.5
box(RX, PY, RW, PH, 'white', ORG_E, 1.8, 2)
t(RX+2, PY+PH-2.6, 'What the model learned (explainability)', 13.5, ORG_E, 'bold')
ch = [('SHAP', 'drivers of "high"', GRN, GRN_E), ('Concept importance', 'grouped columns', ORG, ORG_E), ('Surrogate tree', '81.5% fidelity', YEL, INK)]
cw2 = 21.2
for i, (a, b, fc, ec) in enumerate(ch):
    cx = RX+2 + i*(cw2+1.3); box(cx, PY+PH-12.2, cw2, 6.4, fc, ec, 1.6, 1.2)
    t(cx+cw2/2, PY+PH-8.1, a, 10.5, INK, 'bold', 'center'); t(cx+cw2/2, PY+PH-10.7, b, 8.6, MUTE, ha='center')
box(RX+1.5, PY+1.7, RW-3, PH-15.2, GRY, '#cbd2d9', 1, 1.2)
t(RX+3, PY+PH-15.4, 'Strongest drivers and rules', 10.5, INK, 'bold')
exp = [('Top drivers of "high"', 'hour · lighting · city crime rate'),
       ('Complaint type & severity', '26% of importance'),
       ('City crime context', '20%'),
       ('Time of day', '17%'),
       ('Tree rule for "high"', 'evening/night + lighting ≤ 2'),
       ('', 'and crime rate > 152.7')]
for i, (a, b) in enumerate(exp):
    yy = PY+PH-17.5-i*1.8
    if a: t(RX+3, yy, a, 9.6)
    t(RX+30 if a else RX+32.2, yy, ('→  ' if a else '') + b, 9.6, INK, 'bold')

# bottom-left: validation flow
BX, BY, BW, BH = 6, 5.4, 90, 16.4
box(BX, BY, BW, BH, 'white', INK, 1.8, 2)
t(BX+2, BY+BH-2.2, 'Why validate on unseen cities', 13.5, INK, 'bold')
fl = [('Same-city rows', 'share city values'), ('Random split', 'macro-F1 0.999'), ('Hold out whole cities', 'GroupKFold, 5 folds'), ('Honest score', 'macro-F1 0.779 (1 fold)')]
fw = 19.6
for i, (a, b) in enumerate(fl):
    fx = BX+2 + i*(fw+2.5); hot = i == 3
    box(fx, BY+4.6, fw, 7.6, ORG if hot else YEL, ORG_E if hot else INK, 1.6, 1.2)
    t(fx+fw/2, BY+9.6, a, 9.8, INK, 'bold', 'center'); t(fx+fw/2, BY+6.6, b, 8.8, MUTE, ha='center')
    if i < 3: arrow((fx+fw+0.2, BY+8.4), (fx+fw+2.3, BY+8.4))
t(BX+BW/2, BY+2.0, 'Output: a risk-awareness level (low / medium / high) with probabilities, for scenarios in cities it has not seen', 9.2, MUTE, ha='center')

# bottom-right: limitations
LX, LW = 99.5, 54.5
box(LX, BY, LW, BH, GRY, '#cbd2d9', 1.4, 2)
t(LX+2, BY+BH-2.6, 'Limitations', 13.5, INK, 'bold')
for i, s in enumerate(['Proxy labels: the model learns a rule, not real crime',
                       'Only 34 cities: scores vary by fold (about ±0.05)',
                       'Features chosen on the scoring folds: slightly optimistic',
                       'Not for policing, emergencies or individual scoring']):
    yy = BY+BH-6.0-i*2.9
    ax.add_patch(plt.Circle((LX+3.2, yy), 0.5, color=BLU_E)); t(LX+5, yy, s, 9.6)

t(80, 3.3, 'Tools: Python · pandas · scikit-learn · LightGBM · SHAP · matplotlib / seaborn · Google Colab', 10, MUTE, ha='center')
fig.savefig('figures/project_diagram.png', dpi=130, facecolor='white')
