import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, cm

import argparse

from meanfield import MeanField
from train_acc_util import MNISTtrainer, compute_training_acc


def d_tanh(x):
    """Derivative of tanh."""
    return 1. / np.cosh(x)**2


def simple_plot(x, y):
    plt.plot(x, y)

    plt.xlim(1.0, 4.0)
    plt.ylim(10, 100)
    plt.xlabel('$\sigma_\omega^2$', fontsize=16)
    plt.ylabel('depth scale', fontsize=16)
    plt.show()


def plot(x, y, train_acc_contour=None):
    fontsize = 20
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=int(fontsize / 1.5))
    plt.rc('ytick', labelsize=int(fontsize / 1.5))

    # plot contour
    if train_acc_contour:
        # parse
        xgrid = train_acc_contour['xgrid']
        ygrid = train_acc_contour['ygrid']
        train_acc = train_acc_contour['train_acc']

        # plot contourf
        XGrid, YGrid = np.meshgrid(xgrid, ygrid)
        trian_acc_range = np.linspace(0., 1.0, 11, endpoint=True)

        plt.contourf(XGrid, YGrid, train_acc, trian_acc_range, cmap=cm.PuBu_r)
        plt.colorbar()

    # plot theoretical prediction with different multipliers
    plt.plot(x, y, linewidth=2, linestyle='dashed', color='#8e0000')
    plt.plot(x, y * 2, linewidth=2, linestyle='dashed', color='#c60000')
    plt.plot(x, y * 4.5, linewidth=3, linestyle='dashed', color='red')

    # add text
    plt.text(2.5, 12, r'$\xi_{c^*}$', fontsize=fontsize, color='#8e0000')
    plt.text(2.8, 22, r'2$\xi_{c^*}$', fontsize=fontsize, color='#c60000')
    plt.text(3.0, 30, r'4.5$\xi_{c^*}$', fontsize=fontsize, color='red')

    # setting
    plt.xlim(1.0, 4.0)
    plt.ylim(10, 100)
    plt.xlabel(r'$\sigma_w^2$', fontsize=fontsize)
    plt.ylabel(r'\textit{depth}', fontsize=fontsize)

    # show plot
    plt.tight_layout()
    plt.show()


def exp_depth_scale(filename=None):
    if filename:
        print('Load existing result for depth scale experiment.')
        f = np.load(filename)
        return [f['sws'], f['xi_cs']]

    print('Start depth scale experiment.')

    # Run mean field experiment
    mf = MeanField(np.tanh, d_tanh)

    # set initial condition
    qa0, qb0, c0 = 0.8, 0.8, 0.6

    # set variance of bias and weight
    sb, sws = 0.05, np.linspace(1, 4, 50)

    # set max literation and convergence threshold
    maxL, tol = 300, 1e-6

    # run mean field experiment
    xi_cs = [mf.xi_c(sw, sb, qa0, qb0, c0, maxL, tol) for sw in sws]
    # np.savez('depth_scale', sws=sws,xi_cs=xi_cs)

    return (sws, xi_cs)


def exp_trainability(args, filename=None):
    if filename:
        print('Load existing result for train accuracy.')
        f = np.load(filename)
        return dict(xgrid=f['xgrid'],
                    ygrid=f['ygrid'],
                    train_acc=f['train_acc'])

    print('Start trainability experiment.')

    # setup dataset and hyper-params
    dataset = MNISTtrainer(args.batch_size)
    params = dict(device='cpu' if args.no_cuda else 'cuda',
                  width=args.width, lr=args.lr, num_train=args.num_train,
                  sb=0.05,
     )
    if args.debug: print(params)

    # run experiment in grid coordinate.
    sws = np.linspace(1., 4., args.num_sw)
    depths = np.linspace(10, 100, args.num_depth, dtype=int)
    accs = list()
    for i, depth in enumerate(depths):
        for j, sw in enumerate(sws):
            params['depth'], params['sw'] = depth, sw
            acc = compute_training_acc(dataset, params, debug=args.debug_train_acc)
            accs.append(acc)
            if args.debug: print('({},{})->[d={},sw={}]: \t Train Acc: {:.6f}'.format(i,j,depth,sw,acc))

    acc = np.array(accs).reshape((len(depths), len(sws)))
    # np.savez('train_acc', xgrid=sws,ygrid=depths,train_acc=acc)

    return dict(xgrid=sws,ygrid=depths,train_acc=acc)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth-scale-file', default=None, type=str,
                        help='file name of depth scale data')
    parser.add_argument('--train-acc-file', default=None, type=str,
                        help='file name of train accuracy data')

    # the following arguments are only for trainability experiment.
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-sw', default=90, type=int,
                        help='number of weight variance between [1,4]')
    parser.add_argument('--num-depth', default=91, type=int,
                        help='number of depth between [10,100]')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size for SGD')
    parser.add_argument('--num-train', default=200, type=int,
                        help='number of training steps')
    parser.add_argument('--width', default=300, type=int,
                        help='width of fully-connected layer')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate for SGD')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug the main experiment')
    parser.add_argument('--debug-train-acc', action='store_true', default=False,
                        help='debug the compute_train_acc func')

    args = parser.parse_args()

    # Run experiment.
    sws, xi_cs = exp_depth_scale(args.depth_scale_file)
    train_acc = exp_trainability(args, args.train_acc_file)

    # for simplified figure
    simple_plot(sws, xi_cs)

    # for creating the actual figure in the paper.
    plot(sws, xi_cs, train_acc)
