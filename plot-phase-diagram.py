import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from meanfield import MeanField


def d_tanh(x):
    """Derivative of tanh."""
    return 1. / np.cosh(x)**2


def simple_plot(x, y):
    plt.plot(x, y)
    plt.xlim(0.5, 3)
    plt.ylim(0, 0.25)
    plt.xlabel('$\sigma_\omega^2$', fontsize=16)
    plt.ylabel('$\sigma_b^2$', fontsize=16)

    plt.show()


def plot(x, y):
    fontsize = 12
    plt.figure(figsize=(4, 3.1))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=int(fontsize / 1.5))
    plt.rc('ytick', labelsize=int(fontsize / 1.5))

    # plot critical line
    plt.plot(x, y, linewidth=2, color='black')

    # plot dashed line for sb= 0.05
    x_c = np.interp(0.05, y, x)  # 1.7603915227624916

    line_dict = dict(linewidth=1.5, linestyle='dashed', color='black')
    plt.plot([0.5, x_c], [0.05, 0.05], **line_dict)
    plt.plot([x_c, x_c], [0.00, 0.05], **line_dict)

    # fill ordered and chaotic phase
    plt.fill_betweenx(y, x, 3.0, facecolor='#ffdad3')
    plt.fill_betweenx(y, 0.5, x, facecolor='#d3e4ff')

    # setting
    fontsize = 12
    plt.xlim(0.5, 3)
    plt.ylim(0, 0.25)
    plt.xlabel('$\sigma_\omega^2$', fontsize=fontsize)
    plt.ylabel('$\sigma_b^2$', fontsize=fontsize)

    plt.xlabel(r'$\sigma_w^2$', fontsize=fontsize)
    plt.ylabel(r'$\sigma_b^2$', fontsize=fontsize)

    # add text
    text_dict = dict(fontsize=fontsize,
                     horizontalalignment='center',
                     verticalalignment='center')
    plt.text(1.25, 0.15, r'\textbf{Ordered Phase}', **text_dict)
    plt.text(1.25, 0.125, r'$\max(\chi_{q^*}, \chi_{c^*}) < 1$', **text_dict)
    plt.text(2.475, 0.08, r'\textbf{Chaotic Phase}', **text_dict)
    plt.text(2.475, 0.055, r'$\max(\chi_{q^*}, \chi_{c^*}) > 1$', **text_dict)

    # show plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # run mean field experiment.
    mf = MeanField(np.tanh, d_tanh)

    qrange = np.linspace(1e-5, 2.25, 50)
    sw_sbs = [mf.sw_sb(q, 1.0) for q in qrange]
    sw = [sw_sb[0] for sw_sb in sw_sbs]
    sb = [sw_sb[1] for sw_sb in sw_sbs]

    # for simplified figure
    simple_plot(sw, sb)

    # for creating the actual figure in the paper.
    plot(sw, sb)
