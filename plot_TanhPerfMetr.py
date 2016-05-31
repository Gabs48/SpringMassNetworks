import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from roboTraining.utils import Plot

delta = 0.05
p = np.arange(0.01, 5, delta)
d = np.arange(0, 2, delta)
P, D = np.meshgrid(p, d)

def perfmetr(P,D):
    beta = np.arctanh(1 / np.sqrt(2))
    return np.tanh(beta / P) * np.tanh(beta * D)

M = perfmetr(P, D)

# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
fig, ax = Plot.initPlot()
CS = ax.contour(P, D, M,[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75])
plt.clabel(CS, inline=1, fontsize=10)
Plot.configurePlot(fig, ax,'','',legend = False)
labelsize = 22
ax.set_xlabel(r'$\frac{p}{p_{ref}}$', size = labelsize)
ax.set_ylabel( r'$\frac{d}{d_{ref}}$', size = labelsize)
fig.tight_layout()
plt.show()
Plot.save2eps(fig, 'tanhPerfMetr')
