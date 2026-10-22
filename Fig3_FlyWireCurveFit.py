import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import odr
from scipy.special import zeta

# Load data
UND_CSV = '[...]'
OUT_CSV = '[...]'
df = pd.read_csv(UND_CSV)

syn_counts = df['undirected_syn_count'].to_numpy()
total_synapses = syn_counts.sum()
num_connections = len(syn_counts)

print(f'Total synapses: {total_synapses}')

unique_counts, counts = np.unique(syn_counts, return_counts=True)

sort_idx = np.argsort(unique_counts)
xdata = unique_counts[sort_idx].astype(float)
ydata = counts[sort_idx] / num_connections

# ~~~~~ Curve Fitting via Orthogonal Distance Regression (ODR) ~~~~~ #

# p[0] = A, p[1] = alpha, p[2] = gamma
def log_model(p, x):
    alpha, gamma = p[0], p[1]

    if alpha <= 0 or gamma <= 2:
        return np.ones_like(x) * -np.inf
    return np.log10(alpha**gamma) - np.log10(zeta(gamma, 1.0/alpha) - alpha**gamma) - gamma * np.log10(1.0 + alpha * (10**x))

X_log = np.log10(xdata)
Y_log = np.log10(ydata)

guess_gamma = 2.5                    
guess_alpha = 1.0 / xdata.mean()

initial_guesses = [guess_alpha, guess_gamma]

odr_model = odr.Model(log_model)
odr_data = odr.Data(X_log, Y_log)
fit_odr = odr.ODR(odr_data, odr_model, beta0=initial_guesses)
fit_output = fit_odr.run()

alpha_fit, gamma_fit = fit_output.beta
print(f"Fitted Parameters -> alpha: {alpha_fit:.4f}, gamma: {gamma_fit:.4f}")

# Plotting

mm = 1 / 25.4

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

plt.rcParams['font.size'] = 10

plt.rcParams['axes.labelsize'] = 10  
plt.rcParams['axes.titlesize'] = 10   
plt.rcParams['figure.titlesize'] = 10
plt.rcParams['legend.fontsize'] = 9     
plt.rcParams['legend.title_fontsize'] = 9 

plt.rcParams['xtick.labelsize'] = 7     
plt.rcParams['ytick.labelsize'] = 7

plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

fig, ax = plt.subplots(figsize=(60 * mm, 60 * mm), constrained_layout=True)

ax.scatter(xdata, ydata, color="#00A0CC", s=49/2, zorder=20, clip_on=False, label='Real')

fitx = np.logspace(0, 4, 1000)
fity = ((alpha_fit**gamma_fit)/(zeta(gamma_fit, 1 / alpha_fit) - alpha_fit**gamma_fit)) * (1.0 + alpha_fit * fitx) ** (-gamma_fit)
ax.plot(fitx, fity, 'k-', linewidth=2, zorder=30, label='Model (analytic)', clip_on=True)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Synapses per connection')
ax.set_ylabel('Probability')
ax.set_xlim(1, 10**4)
ax.set_ylim(10**(-8), 1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
ax.legend(frameon=False, loc='upper right')

calculated_pbps = alpha_fit * (gamma_fit-1) 
calculated_ptps = alpha_fit * (gamma_fit-2)
calculated_pbpt = calculated_pbps/calculated_ptps

ax.text(
    0.05, 0.05, 
    f'$r_b =$ {calculated_pbps:.2f}\n$r_t = ${calculated_ptps:.2f}\n$r_b/r_t = ${calculated_pbpt:.2f}', 
    transform=ax.transAxes, 
    verticalalignment='bottom', 
    horizontalalignment='left',
    fontsize=9
)

plt.savefig(OUT_CSV, format='svg')
