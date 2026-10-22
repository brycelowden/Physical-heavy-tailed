import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import odr
from scipy.special import zeta

df = pd.read_csv('C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/C. Elegans Data/Undirected_Celegans.csv')

#Create numpy array of connection strengths
strength_values = df['synapse'].values

unique_strengths, counts = np.unique_counts(strength_values)

xdata = unique_strengths.tolist()

total_connections = counts.sum()
ydata = (counts / total_connections).tolist()

print(f'\nTotal connections: {total_connections}')

print(f'\nTotal synapses: {sum(strength_values)}')

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
guess_alpha = 1.0 / np.mean(xdata)     # Scale factor guess

initial_guesses = [guess_alpha, guess_gamma]

# Set up and execute ODR
odr_model = odr.Model(log_model)
odr_data = odr.Data(X_log, Y_log)
fit_odr = odr.ODR(odr_data, odr_model, beta0=initial_guesses)
fit_output = fit_odr.run()

# Extract optimized parameters
alpha_fit, gamma_fit = fit_output.beta
print(f"Fitted Parameters -> alpha: {alpha_fit:.4f}, gamma: {gamma_fit:.4f}")

#------------------------------------
#Plotting
#------------------------------------

mm = 1 / 25.4  # 1 mm in inches

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 2. Set the Base Font Size (This acts as the default fallback)
plt.rcParams['font.size'] = 10

# 3. Explicitly force Size 10 for all major text elements
plt.rcParams['axes.labelsize'] = 10       # X and Y axis labels
plt.rcParams['axes.titlesize'] = 10       # Subplot titles
plt.rcParams['figure.titlesize'] = 10     # Global figure suptitles
plt.rcParams['legend.fontsize'] = 7      # Legend text
plt.rcParams['legend.title_fontsize'] = 10 # Legend title

# 4. Explicitly restrict Tick Numbers to Size 7
plt.rcParams['xtick.labelsize'] = 7       # X-axis tick numbers
plt.rcParams['ytick.labelsize'] = 7       # Y-axis tick numbers

plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

fitx = np.logspace(0,4,1000)
fity = ((alpha_fit**gamma_fit)/(zeta(gamma_fit, 1 / alpha_fit) - alpha_fit**gamma_fit)) * (1.0 + alpha_fit * fitx) ** (-gamma_fit)

fig, ax = plt.subplots(figsize=(60 * mm,60 * mm), constrained_layout=True)
ax.scatter(xdata,ydata, color = "#ffc600ff",s = 49/2, zorder = 20,clip_on=False,label = 'Real')
ax.plot(fitx,fity,'k-',linewidth = 2, zorder = 30,label = 'Model (analytic)',clip_on=True)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Synapses per connection')
ax.set_ylabel('Probability')
ax.set_xlim(1,10**2)
ax.set_ylim(10**(-4),1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
#ax.legend(frameon = False, loc='lower left')

calculated_pbps = alpha_fit * (gamma_fit-1) #pb/ps = B(gamma - 1)
calculated_ptps = alpha_fit * (gamma_fit-2) #pt/ps = B(gamma - 2)
calculated_pbpt = calculated_pbps/calculated_ptps

ax.text(
    0.05, 0.05, 
    f'$r_b = ${calculated_pbps:.2f}\n$r_t = ${calculated_ptps:.2f}\n$r_b/r_t = ${calculated_pbpt:.2f}', 
    transform=ax.transAxes, 
    verticalalignment='bottom', 
    horizontalalignment='left',
    fontsize=9
)

plt.savefig('C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/Figure Files/Figure 3/CElegans_Undirected.svg',format = 'svg')