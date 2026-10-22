import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import odr
import matplotlib.ticker as ticker
from scipy.special import zeta
import math

df = pd.read_csv('C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/BANCData/BANCUndirectedConnections.csv')
cs = df['undirected_syn_count'].value_counts()

xdata = []
ydata = []

for i in cs.index:
    if i >= 3:
        xdata.append(i)
        ydata.append(int(cs.loc[i]))

numConnections = sum(ydata)
for i in range(len(ydata)):
    ydata[i] = ydata[i] / numConnections

# ~~~~~ Curve Fitting via Orthogonal Distance Regression (ODR) ~~~~~ #

# Define the model in log-space to minimize visual/geometric distance
# p[0] = A, p[1] = alpha, p[2] = gamma
def log_model(p, x):
    alpha, gamma = p[0], p[1]
    # Guard against invalid negative values chosen during optimization iterations
    if alpha <= 0 or gamma <= 2:
        return np.ones_like(x) * -np.inf
    return np.log10(alpha**gamma) - np.log10(zeta(gamma, 1.0/alpha) - alpha**gamma - (1/alpha + 1)**(-gamma) - (1/alpha + 2)**(-gamma)) - gamma * np.log10(1.0 + alpha * (10**x))

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

fitx = np.logspace(0, 4, 1000)
fity = ((alpha_fit**gamma_fit)/(zeta(gamma_fit, 1 / alpha_fit) - alpha_fit**gamma_fit - (1/alpha_fit + 1)**(-gamma_fit) - (1/alpha_fit + 2)**(-gamma_fit))) * (1.0 + alpha_fit * fitx) ** (-gamma_fit)

fig, ax = plt.subplots(figsize=(60 * mm,60 * mm), constrained_layout=True)
ax.scatter(xdata,ydata, color = "#6646daff",s = 49/2, zorder = 20,clip_on=False,label = 'Real')
ax.plot(fitx,fity,'k-',linewidth = 2, zorder = 30,label = 'Model (analytic)',clip_on=True)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Synapses per connection')
ax.set_ylabel('Probability')
ax.set_xlim(3,2*10**3) #BANC ignores connections with 1 or 2 synapses
ax.set_ylim(10**(-7),1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 

# ax.xaxis.set_minor_locator(ticker.NullLocator())

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

plt.savefig('C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/Figure Files/Figure 3/BANC_UndirectedCosyne.svg',format = 'svg')