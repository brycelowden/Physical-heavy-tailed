import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import odr
from scipy.special import zeta

# Load data
UND_CSV = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Data/FlyWireUndCons.csv'
OUT_CSV = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/Figure Files/Figure 3/FAFB_Undirected.svg'
df = pd.read_csv(UND_CSV)

# Calculate basic statistics
syn_counts = df['undirected_syn_count'].to_numpy()
total_synapses = syn_counts.sum()
num_connections = len(syn_counts)

print(f'Total synapses: {total_synapses}')

# ~~~~~ Efficient Data Aggregation with NumPy ~~~~~ #
unique_counts, counts = np.unique(syn_counts, return_counts=True)

# Sort in ascending order of synapse counts
sort_idx = np.argsort(unique_counts)
xdata = unique_counts[sort_idx].astype(float)
ydata = counts[sort_idx] / num_connections

# ~~~~~ Curve Fitting via Orthogonal Distance Regression (ODR) ~~~~~ #

# Define the model in log-space to minimize visual/geometric distance
# p[0] = A, p[1] = alpha, p[2] = gamma
def log_model(p, x):
    alpha, gamma = p[0], p[1]
    # Guard against invalid negative values chosen during optimization iterations
    if alpha <= 0 or gamma <= 2:
        return np.ones_like(x) * -np.inf
    return np.log10(alpha**gamma) - np.log10(zeta(gamma, 1.0/alpha) - alpha**gamma) - gamma * np.log10(1.0 + alpha * (10**x))

# Transform data to log-space for the visual optimizer
X_log = np.log10(xdata)
Y_log = np.log10(ydata)

# Formulate smart initial guesses based on your data shape
guess_gamma = 2.5                    # Standard power-law tail fallback
guess_alpha = 1.0 / xdata.mean()     # Scale factor guess

initial_guesses = [guess_alpha, guess_gamma]

# Set up and execute ODR
odr_model = odr.Model(log_model)
odr_data = odr.Data(X_log, Y_log)
fit_odr = odr.ODR(odr_data, odr_model, beta0=initial_guesses)
fit_output = fit_odr.run()

# Extract optimized parameters
alpha_fit, gamma_fit = fit_output.beta
print(f"Fitted Parameters -> alpha: {alpha_fit:.4f}, gamma: {gamma_fit:.4f}")

# ~~~~~ Plotting & Formatting ~~~~~ #

mm = 1 / 25.4  # 1 mm in inches

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 2. Set the Base Font Size (This acts as the default fallback)
plt.rcParams['font.size'] = 10

# 3. Explicitly force Size 10 for all major text elements
plt.rcParams['axes.labelsize'] = 10        # X and Y axis labels (10)
plt.rcParams['axes.titlesize'] = 10       # Subplot titles (10)
plt.rcParams['figure.titlesize'] = 10     # Global figure suptitles
plt.rcParams['legend.fontsize'] = 9       # Legend text
plt.rcParams['legend.title_fontsize'] = 9 # Legend title

# 4. Explicitly restrict Tick Numbers to Size 7
plt.rcParams['xtick.labelsize'] = 7       # X-axis tick numbers
plt.rcParams['ytick.labelsize'] = 7       # Y-axis tick numbers

plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

# (60mm, 60mm)
fig, ax = plt.subplots(figsize=(60 * mm, 60 * mm), constrained_layout=True)

# Scatter real data points
ax.scatter(xdata, ydata, color="#00A0CC", s=49/2, zorder=20, clip_on=False, label='Real')

# Generate smooth visual curve using the fitted parameters
fitx = np.logspace(0, 4, 1000)
fity = ((alpha_fit**gamma_fit)/(zeta(gamma_fit, 1 / alpha_fit) - alpha_fit**gamma_fit)) * (1.0 + alpha_fit * fitx) ** (-gamma_fit)
ax.plot(fitx, fity, 'k-', linewidth=2, zorder=30, label='Model (analytic)', clip_on=True)

# Scale and limits
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Synapses per connection')
ax.set_ylabel('Probability')
ax.set_xlim(1, 10**4)
ax.set_ylim(10**(-8), 1)

# Aesthetics
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
ax.legend(frameon=False, loc='upper right')

# Text annotations (Note: ensure calculated_pbps, calculated_ptps, etc., are defined above in your full script)
# Using placeholders here to prevent NameErrors if missing

calculated_pbps = alpha_fit * (gamma_fit-1) #pb/ps = B(gamma - 1)
calculated_ptps = alpha_fit * (gamma_fit-2) #pt/ps = B(gamma - 2)
calculated_pbpt = calculated_pbps/calculated_ptps

# fontsize=9
ax.text(
    0.05, 0.05, 
    f'$r_b =$ {calculated_pbps:.2f}\n$r_t = ${calculated_ptps:.2f}\n$r_b/r_t = ${calculated_pbpt:.2f}', 
    transform=ax.transAxes, 
    verticalalignment='bottom', 
    horizontalalignment='left',
    fontsize=9
)

# Save figure
plt.savefig(OUT_CSV, format='svg')