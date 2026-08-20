import multiprocessing as mp
import numpy as np
import numpy.polynomial.polynomial as npoly
import matplotlib.pyplot as plt

from scipy.special import logsumexp
from numpy.polynomial.legendre import leggauss

import emcee
import corner


class PolynomialModel:

    def __init__(self, x, y, y_err, x_err, xy_err, order=1, n_quad=32):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.y_err = np.asarray(y_err, dtype=float)
        self.x_err = np.asarray(x_err, dtype=float)
        self.xy_err = np.asarray(xy_err, dtype=float)

        self.order = order
        self.ndim = order + 4

        self.xmin = np.min(self.x)
        self.xmax = np.max(self.x)

        # ============================================================
        # GAUSS-LEGENDRE QUADRATURE
        # ============================================================
        self.n_quad = n_quad

        # Nodes z_k and weights w_k on [-1, 1]
        self.z_quad, self.w_quad = leggauss(n_quad)

        # Convert quadrature nodes from z in [-1, 1] to physical
        # latent x_t values:
        #   x_t = (xmax + xmin)/2 + (xmax - xmin)/2 * z_t
        xmid = 0.5 * (self.xmax + self.xmin)
        xhalf = 0.5 * (self.xmax - self.xmin)
        self.x_quad = xmid + xhalf * self.z_quad

    # ================================================================
    # POLYNOMIAL MODEL
    # ================================================================

    def poly_model(self, x, coeffs):
        """p(x) = a_0 + a_1 x + ... + a_n x^n"""
        return npoly.polyval(x, coeffs)

    # ================================================================
    # MARGINALIZED LOG LIKELIHOOD
    # ================================================================

    def ln_likelihood(self, pars):
        # ------------------------------------------------------------
        # MODEL PARAMETERS
        # ------------------------------------------------------------
        coeffs = pars[:self.order + 1]
        sigma_x_int = pars[self.order + 1]
        sigma_y_int = pars[self.order + 2]
        sigma_xy_int = pars[self.order + 3]

        # ------------------------------------------------------------
        # TOTAL COVARIANCE
        #
        #   V_i = [ Sigma_x,i   Sigma_xy,i ]
        #         [ Sigma_xy,i  Sigma_y,i  ]
        #
        #   Sigma_x,i  = sigma_x,obs^2  + sigma_x,int^2
        #   Sigma_y,i  = sigma_y,obs^2  + sigma_y,int^2
        #   Sigma_xy,i = sigma_xy,obs   + sigma_xy,int
        # ------------------------------------------------------------
        Sigma_x = self.x_err**2 + sigma_x_int**2
        Sigma_y = self.y_err**2 + sigma_y_int**2
        Sigma_xy = self.xy_err + sigma_xy_int

        # Determinant of V_i
        detV = Sigma_x * Sigma_y - Sigma_xy**2

        # Covariance matrix must be positive definite
        if np.any(~np.isfinite(detV)) or np.any(detV <= 0):
            return -np.inf

        # ------------------------------------------------------------
        # POLYNOMIAL AT QUADRATURE POINTS
        #   x_t,k -> p(x_t,k)      shape: (n_quad,)
        # ------------------------------------------------------------
        y_quad = self.poly_model(self.x_quad, coeffs)

        # ------------------------------------------------------------
        # RESIDUALS (every data point i, every quadrature point k)
        #   dx[i, k] = x_i - x_t,k
        #   dy[i, k] = y_i - p(x_t,k)      shapes: (N_data, N_quad)
        # ------------------------------------------------------------
        dx = self.x[:, None] - self.x_quad[None, :]
        dy = self.y[:, None] - y_quad[None, :]

        # ------------------------------------------------------------
        # M_i(z_k) = d^T V^-1 d, evaluated at every quadrature node
        #
        #   M = [ Sigma_y * dx^2 - 2 Sigma_xy * dx * dy
        #         + Sigma_x * dy^2 ] / det(V)
        # ------------------------------------------------------------
        M = (
            Sigma_y[:, None] * dx**2
            - 2.0 * Sigma_xy[:, None] * dx * dy
            + Sigma_x[:, None] * dy**2
        ) / detV[:, None]

        # ------------------------------------------------------------
        # MARGINALIZATION OVER x_t
        #
        #   p(x_t) = 1 / (xmax - xmin)   over [xmin, xmax]
        #
        # After transforming x_t -> z_t:
        #   I_i = 1/2 * integral[-1, 1] exp[-M_i(z_t)/2] dz_t
        #
        # Gauss-Legendre:
        #   I_i ~= 1/2 * sum_k w_k exp[-M_i(z_k)/2]
        #
        # We compute log[w_k * exp(-M/2)] rather than exp(-M/2)
        # directly -- much more numerically stable during MCMC.
        # ------------------------------------------------------------
        log_integrand = -0.5 * M + np.log(self.w_quad)[None, :]

        # log(I_i) = log(1/2) + log[ sum_k w_k exp(-M_i(z_k)/2) ]
        log_integral = np.log(0.5) + logsumexp(log_integrand, axis=1)

        # ------------------------------------------------------------
        # PER-DATA-POINT LOG LIKELIHOOD
        #   ln L_i = -ln(2 pi) - 1/2 ln(det V_i) + ln(I_i)
        # ------------------------------------------------------------
        logL_i = -np.log(2.0 * np.pi) - 0.5 * np.log(detV) + log_integral

        # TOTAL LOG LIKELIHOOD:  ln L = sum_i ln L_i
        return np.sum(logL_i)

    # ================================================================
    # PRIOR
    # ================================================================

    def ln_prior(self, pars):
        coeffs = pars[:self.order + 1]
        sigma_x_int = pars[self.order + 1]
        sigma_y_int = pars[self.order + 2]
        sigma_xy_int = pars[self.order + 3]

        # Polynomial coefficient bounds
        if np.any(np.abs(coeffs) > 1e6):
            return -np.inf

        # Intrinsic scatter must be non-negative
        if sigma_x_int <= 0 or sigma_x_int > 1e6:
            return -np.inf

        if sigma_y_int <= 0 or sigma_y_int > 1e6:
            return -np.inf

        # Check TOTAL covariance matrix:
        #   V_i = [ Sigma_x   Sigma_xy ]
        #         [ Sigma_xy  Sigma_y  ]
        # must have det(V_i) > 0
        Sigma_x = self.x_err**2 + sigma_x_int**2
        Sigma_y = self.y_err**2 + sigma_y_int**2
        Sigma_xy = self.xy_err + sigma_xy_int
        detV = Sigma_x * Sigma_y - Sigma_xy**2

        if np.any(detV <= 0):
            return -np.inf

        return 0.0

    # ================================================================
    # POSTERIOR
    # ================================================================

    def ln_posterior(self, pars):
        lp = self.ln_prior(pars)
        if not np.isfinite(lp):
            return -np.inf

        ll = self.ln_likelihood(pars)
        if not np.isfinite(ll):
            return -np.inf

        return lp + ll

    # ================================================================
    # CALLABLE
    # ================================================================

    def __call__(self, pars):
        return self.ln_posterior(pars)


# ====================================================================
# MCMC FIT
# ====================================================================

def hoggfit(x, y, xerr=None, yerr=None, xyerr=None, order=1,
            nwalkers=32, nsteps=10000, ncores=None, n_quad=32):

    rng = np.random.default_rng()

    # ------------------------------------------------------------
    # Convert inputs to arrays
    # ------------------------------------------------------------
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xerr = np.zeros_like(x) if xerr is None else np.asarray(xerr, dtype=float)
    yerr = np.zeros_like(y) if yerr is None else np.asarray(yerr, dtype=float)
    xyerr = np.zeros_like(x) if xyerr is None else np.asarray(xyerr, dtype=float)

    # ------------------------------------------------------------
    # Remove invalid data
    # ------------------------------------------------------------
    mask = (
        np.isfinite(x) & np.isfinite(y)
        & np.isfinite(xerr) & np.isfinite(yerr) & np.isfinite(xyerr)
    )
    x, y, xerr, yerr, xyerr = x[mask], y[mask], xerr[mask], yerr[mask], xyerr[mask]

    # ------------------------------------------------------------
    # Sort by x
    # ------------------------------------------------------------
    idx = np.argsort(x)
    x, y, xerr, yerr, xyerr = x[idx], y[idx], xerr[idx], yerr[idx], xyerr[idx]

    # ------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------
    model = PolynomialModel(x, y, yerr, xerr, xyerr, order=order, n_quad=n_quad)
    ndim = model.ndim

    # ------------------------------------------------------------
    # Initial polynomial fit
    # ------------------------------------------------------------
    coeffs0 = npoly.polyfit(x, y, order)
    y0 = model.poly_model(x, coeffs0)
    residual = y - y0
    sigma0 = max(np.std(residual), 1e-6)

    # ------------------------------------------------------------
    # Initial intrinsic scatter
    # ------------------------------------------------------------
    sigma_x_int0 = max(0.01 * np.std(x), 1e-6)
    sigma_y_int0 = sigma0
    sigma_xy_int0 = 0.0

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    print("\n------------------------------------------")
    print("Initial fit diagnostics")
    print("------------------------------------------")
    print("Polynomial coefficients:", coeffs0)
    print("Initial residual scatter:", sigma0)
    print("Initial RMSE:", np.sqrt(np.mean(residual**2)))

    initial_params_test = np.concatenate(
        [coeffs0, [sigma_x_int0, sigma_y_int0, sigma_xy_int0]]
    )
    print("Initial log posterior:", model.ln_posterior(initial_params_test))
    print("Number of quadrature points:", n_quad)
    print("------------------------------------------\n")

    # ------------------------------------------------------------
    # Initial fit plot
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="black", label="Data")

    x_plot = np.linspace(x.min(), x.max(), 1000)
    y_plot = model.poly_model(x_plot, coeffs0)
    plt.plot(x_plot, y_plot, color="red", lw=2, label="Initial polynomial")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Initialize walkers
    # ------------------------------------------------------------
    initial_params = np.zeros((nwalkers, ndim))

    # Polynomial coefficients
    for k in range(order + 1):
        scale = max(abs(coeffs0[k]) * 1e-3, 1e-6)
        initial_params[:, k] = coeffs0[k] + scale * rng.standard_normal(nwalkers)

    # Intrinsic x scatter
    initial_params[:, order + 1] = np.abs(
        sigma_x_int0 * (1.0 + 0.05 * rng.standard_normal(nwalkers))
    )

    # Intrinsic y scatter
    initial_params[:, order + 2] = np.abs(
        sigma_y_int0 * (1.0 + 0.05 * rng.standard_normal(nwalkers))
    )

    # Intrinsic covariance
    sx = initial_params[:, order + 1]
    sy = initial_params[:, order + 2]
    max_covariance = sx * sy

    initial_params[:, order + 3] = (
        0.1 * max_covariance * rng.standard_normal(nwalkers)
    )
    initial_params[:, order + 3] = np.clip(
        initial_params[:, order + 3], -0.99 * max_covariance, 0.99 * max_covariance
    )

    # ------------------------------------------------------------
    # Multiprocessing
    # ------------------------------------------------------------
    if ncores is None:
        ncores = mp.cpu_count()

    # ------------------------------------------------------------
    # MCMC
    # ------------------------------------------------------------
    with mp.Pool(processes=ncores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, model.ln_posterior, pool=pool
        )
        sampler.run_mcmc(initial_params, nsteps, progress=True)

    return sampler


# ====================================================================
# ANALYSIS
# ====================================================================

def analyze_hoggfit(sampler, x, y, order=1, intrinsic_true=None,
                     truths=None, burnin=250, thin=10, nsamp=200):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # ------------------------------------------------------------
    # Extract chain
    # ------------------------------------------------------------
    chain = sampler.get_chain()
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    # ------------------------------------------------------------
    # Separate parameters
    # ------------------------------------------------------------
    poly_samples = samples[:, :order + 1]
    sigma_x_samples = samples[:, order + 1]
    sigma_y_samples = samples[:, order + 2]
    sigma_xy_samples = samples[:, order + 3]

    full_samples = np.column_stack(
        (poly_samples, sigma_x_samples, sigma_y_samples, sigma_xy_samples)
    )

    # ------------------------------------------------------------
    # Trace plots
    # ------------------------------------------------------------
    labels = [rf"$a_{{{i}}}$" for i in range(order, -1, -1)]
    labels += [r"$\sigma_{\rm int,x}$", r"$\sigma_{\rm int,y}$", r"$\sigma_{\rm int,xy}$"]

    fig, axes = plt.subplots(
        full_samples.shape[1], 1,
        figsize=(10, 2.2 * full_samples.shape[1]), sharex=True
    )
    if full_samples.shape[1] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(chain[:, :, i], color="black", alpha=0.25, lw=0.5)
        ax.set_ylabel(labels[::-1][i])

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Corner plot
    # ------------------------------------------------------------
    truths_plot = None
    if truths is not None:
        truths_plot = list(truths)
        if intrinsic_true is not None:
            truths_plot.extend(list(intrinsic_true))

    corner_labels = [rf"$a_{{{i}}}$" for i in range(order + 1)]
    corner_labels += [r"$\sigma_{\rm int,x}$", r"$\sigma_{\rm int,y}$", r"$\sigma_{\rm int,xy}$"]

    fig = corner.corner(
        samples, labels=corner_labels, truths=truths_plot,
        show_titles=True, title_fmt=".3g",
        quantiles=[0.16, 0.50, 0.84], fill_contours=True,
        smooth=1.0, smooth1d=1.0,
        levels=(0.393, 0.865, 0.989), plot_datapoints=True
    )
    fig.set_size_inches(9, 9)
    plt.show()

    # ------------------------------------------------------------
    # Posterior polynomial curves
    # ------------------------------------------------------------
    rng = np.random.default_rng()
    inds = rng.choice(len(poly_samples), size=min(nsamp, len(poly_samples)), replace=False)

    x_dense = np.linspace(x.min(), x.max(), 1000)
    curves = np.array([npoly.polyval(x_dense, poly_samples[i]) for i in inds])

    lower, upper = np.percentile(curves, [16, 84], axis=0)
    median_curve = np.median(curves, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, color="black", zorder=1)
    ax.fill_between(x_dense, lower, upper, color="orangered", alpha=0.3, label=r"1$\sigma$ CI")
    ax.plot(x_dense, median_curve, color="orangered", lw=2, label="Median model")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid(alpha=0.15)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Parameter correlation matrix
    # ------------------------------------------------------------
    corr = np.corrcoef(samples, rowvar=False)

    corr_labels = [rf"$a_{{{i}}}$" for i in range(order + 1)]
    corr_labels += [r"$\sigma_{\rm int,x}$", r"$\sigma_{\rm int,y}$", r"$\sigma_{\rm int,xy}$"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_xticks(np.arange(len(corr_labels)))
    ax.set_yticks(np.arange(len(corr_labels)))
    ax.set_xticklabels(corr_labels, rotation=45, ha="right")
    ax.set_yticklabels(corr_labels)

    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Posterior Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Posterior quantiles
    # ------------------------------------------------------------
    q16, q50, q84 = np.percentile(full_samples, [16, 50, 84], axis=0)

    print("\n------------------------------------------")
    print("Posterior Polynomial Coefficients")
    print("------------------------------------------")
    for power in range(order, -1, -1):
        i = power
        print(
            f"a_{power:<2d} = {q50[i]:.6e} "
            f"(+{q84[i] - q50[i]:.6e}, -{q50[i] - q16[i]:.6e})"
        )

    # ------------------------------------------------------------
    # Polynomial equation
    # ------------------------------------------------------------
    print("\n------------------------------------------")
    print("Polynomial Equation")
    print("------------------------------------------")

    equation = ""
    for power in range(order, -1, -1):
        c = q50[power]
        if power > 1:
            equation += f"{c:.6e}x^{power}"
        elif power == 1:
            equation += f"{c:.6e}x"
        else:
            equation += f"{c:.6e}"
        if power > 0:
            equation += " + "

    print("y =", equation)

    # ------------------------------------------------------------
    # Intrinsic scatter
    # ------------------------------------------------------------
    print("\n------------------------------------------")
    print("Intrinsic Scatter Covariance")
    print("------------------------------------------")

    sx_q16, sx_q50, sx_q84 = np.percentile(sigma_x_samples, [16, 50, 84])
    sy_q16, sy_q50, sy_q84 = np.percentile(sigma_y_samples, [16, 50, 84])
    sxy_q16, sxy_q50, sxy_q84 = np.percentile(sigma_xy_samples, [16, 50, 84])

    print(f"sigma_int_x  = {sx_q50:.6e} (+{sx_q84 - sx_q50:.6e}, -{sx_q50 - sx_q16:.6e})")
    print(f"sigma_int_y  = {sy_q50:.6e} (+{sy_q84 - sy_q50:.6e}, -{sy_q50 - sy_q16:.6e})")
    print(f"sigma_int_xy = {sxy_q50:.6e} (+{sxy_q84 - sxy_q50:.6e}, -{sxy_q50 - sxy_q16:.6e})")

    intrinsic_covariance = np.array([[sx_q50**2, sxy_q50], [sxy_q50, sy_q50**2]])

    print("\n------------------------------------------")
    print("Median Intrinsic Covariance Matrix")
    print("------------------------------------------")
    print(intrinsic_covariance)

    # ------------------------------------------------------------
    # Posterior covariance matrix
    # ------------------------------------------------------------
    covariance_matrix = np.cov(full_samples, rowvar=False)

    print("\n------------------------------------------")
    print("Posterior Parameter Covariance Matrix")
    print("------------------------------------------")
    print(covariance_matrix)

    return [q16, q50, q84], covariance_matrix, intrinsic_covariance