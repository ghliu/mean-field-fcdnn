import numpy as np
import warnings
from scipy.integrate import quad, dblquad


def gauss_density(x, mu, sigma):
    """Density of Gaussian N(mu,sigma)."""
    k = 1. / (sigma * np.sqrt(2 * np.pi))
    s = -1.0 / (2 * sigma * sigma)
    return k * np.exp(s * (x - mu) * (x - mu))


class MeanField(object):
    def __init__(self, f, df):
        """
        Args:
            f: activation function
            df: derivative of f.
        """
        self.f = f
        self.df = df

    def _qmap_density(self, h, q):
        """Compute the density function of the q-map."""
        return gauss_density(h, 0., np.sqrt(q)) * (self.f(h)**2)

    def _qab_map_density(self, y, x, qaa, qbb, c):
        """
        Compute the density function of the q_{ab}-map.

        Args:
            y: normalized prob. for sample b
            x: normalized prob. for sample a
            qaa: variance of sample a
            qaa: variance of sample b
            c: correlation between samples a and b
        """

        # For numerical stability, we sample from variance=1 bivariate
        # Gaussian and rescale it afterwards.
        y_x = (c * x + np.sqrt(1. - c * c) * y)  # Conditional of y on x.
        ha = np.sqrt(qaa) * x
        hb = np.sqrt(qbb) * y_x
        return gauss_density(x, 0., 1.) * self.f(ha) * \
                gauss_density(y, 0., 1.) * self.f(hb)

    def _chi_c1_density(self, h, q):
        """Compute the density function of chi_c=1."""
        return gauss_density(h, 0., np.sqrt(q)) * (self.df(h)**2)

    def _chi_c_density(self, y, x, q, c):
        """
        Compute the density function of chi_c for arbitrary value of c.

        Args:
            y: normalized prob. for sample b
            x: normalized prob. for sample a
            q: fixed point variance (i.e. q_star)
            c: fixed point correlation (i.e. c_star)
        """

        # For numerical stability, we sample from variance=1 bivariate
        # Gaussian and rescale it afterwards.
        y_x = (c * x + np.sqrt(1. - c * c) * y)  # Conditional of y on x.
        ha = np.sqrt(q) * x
        hb = np.sqrt(q) * y_x
        return gauss_density(x, 0., 1.) * self.df(ha) * \
                gauss_density(y, 0., 1.) * self.df(hb)

    def qmap(self, sw, sb, q):
        """
        Compute the single-sample variance q_{t+1} by propagating q_t to
        the next layer.
        """
        return sb + sw * quad(self._qmap_density, -np.inf, np.inf, args=(q))[0]

    def qab_map(self, sw, sb, qa, qb, c):
        """
        Compute the pair-sample variance q^{(a,b)}_{t+1} by propagating
        q^{(a,b)}_t to the next layer.
        """
        return sb + sw * dblquad(self._qab_map_density,
                                 -np.inf,
                                 np.inf,
                                 lambda x: -np.inf,
                                 lambda x: np.inf,
                                 args=(qa, qb, c))[0]

    def cmap(self, sw, sb, qa, qb, c):
        """
        Compute the correlation c_{t+1} by propagating c_t to the next
        layer.
        """
        qa_nxt = self.qmap(sw, sb, qa)
        qb_nxt = self.qmap(sw, sb, qb)
        qab_nxt = self.qab_map(sw, sb, qa, qb, c)

        return (qab_nxt / np.sqrt(qa_nxt * qb_nxt), qa_nxt, qb_nxt)

    def q_star(self, sw, sb, q0, maxL=50, tol=None):
        """
        Compute the fixed point q^{*}.

        Args:
            sw: variance of initialized weight
            sb: variance of initialized bias
            q0: initial variance
            maxL: maximum iteration
            tol: threshold for early termination when converges.

        Return:
            the fixed point variance q^{*}
            all variance sequence from q0 to q^{*}
        """
        qs = [q0]
        for l in range(maxL - 1):
            qs += [self.qmap(sw, sb, qs[-1])]
            if tol and np.abs(qs[-1] - qs[-2]) < tol:
                break
        return (qs[-1], qs)

    def c_star(self, sw, sb, qa0, qb0, c0, maxL=300, tol=None):
        """
        Compute the fixed point c^{*}.

        Args:
            sw: variance of initialized weight
            sb: variance of initialized bias
            qa0: initial variance for sample a
            qb0: initial variance for sample b
            c0: initial correlation
            maxL: maximum iteration
            tol: threshold for early termination when converges.

        Return:
            the fixed point correlation c^{*}
            all correlation sequence from c0 to c^{*}
        """
        qa, qb, c = qa0, qb0, c0
        cs = [c0]
        for l in range(maxL - 1):
            c, qa, qb = self.cmap(sw, sb, qa, qb, c)
            cs.append(c)

            dc = np.abs(cs[-1] - cs[-2])
            if tol and dc < tol:
                break

        return (cs[-1], cs)

    def sw_sb(self, q, chi1):
        """
        Compute the critical line. Set chi1=1, return variances of weight and
        bias on the critical line with qstar=q.
        """
        with warnings.catch_warnings():
            # silence underflow and overflow warning
            warnings.simplefilter("ignore")
            sw = chi1 / quad(self._chi_c1_density, -np.inf, np.inf, args=(q))[0]
            sb = q - sw * quad(self._qmap_density, -np.inf, np.inf,
                               args=(q))[0]
        return sw, sb

    def xi_c(self, sw, sb, qa0, qb0, c0, maxL=300, tol=None):
        q_star, _ = self.q_star(sw, sb, qa0)
        c_star, _ = self.c_star(sw, sb, qa0, qb0, c0, maxL, tol)

        chi_c = sw * dblquad(self._chi_c_density,
                           -np.inf,
                           np.inf,
                           lambda x: -np.inf,
                           lambda x: np.inf,
                           args=(q_star, c_star))[0]
        return 1. / (-np.log(chi_c))
