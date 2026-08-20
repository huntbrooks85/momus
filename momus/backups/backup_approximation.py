import multiprocessing as mp
import numpy as np
import numpy.polynomial.polynomial as npoly
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.special import factorial, comb
import emcee
import corner


class PolynomialModel:
    def __init__(self, x, y, y_err, x_err, xy_err, order=1):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.y_err = np.asarray(y_err, dtype=float)
        self.x_err = np.asarray(x_err, dtype=float)
        self.xy_err = np.asarray(xy_err, dtype=float)
        self.order = order
        self.ndim = order + 4

    def poly_model(self, x, coeffs):
        return npoly.polyval(x, coeffs)

    def poly_derivative(self, x, coeffs, m=1):
        if m > self.order:
            return np.zeros_like(x, dtype=float)
        return npoly.polyval(x, npoly.polyder(coeffs, m=m))

    def _effective_variance(self, x, coeffs, var_x, var_y, cov_xy, observed_residual):
        """
        Delta_i = (y_i - P(x_i)) - sum_{m=1}^{floor(n/2)} P^(2m)(x_i)/(2^m m!) * Vx^m

        Sigma_i^2 = Vy
            - 2*Cxy * sum_{m=0}^{floor((n-1)/2)} P^(2m+1)(x)/(2^m m!) * Vx^m

            + sum_{m=0}^{floor((n-1)/2)} sum_{l=0}^{floor((n-1)/2)}
                  [1/(2^{m+l+1} (m+l+1)!)] * C(2m+2l+2, 2m+1)
                  * P^(2m+1)(x) * P^(2l+1)(x) * Vx^{m+l+1}

            + sum_{m=1}^{floor(n/2)} sum_{l=1}^{floor(n/2)}
                  [1/(2^{m+l} (m+l)!)] * [C(2m+2l, 2m) - C(m+l, m)]
                  * P^(2m)(x) * P^(2l)(x) * Vx^{m+l}
        """
        N = self.order
        x = np.asarray(x, dtype=float)
        var_x = np.asarray(var_x, dtype=float)
        var_y = np.asarray(var_y, dtype=float)
        cov_xy = np.asarray(cov_xy, dtype=float)

        # ---- Delta_i ----
        mean_correction = np.zeros_like(x, dtype=float)
        for m in range(1, N // 2 + 1):
            Pm = self.poly_derivative(x, coeffs, m=2 * m)
            mean_correction += Pm / (2.0**m * factorial(m)) * var_x**m

        Delta = observed_residual - mean_correction

        # ---- Sigma_i^2 ----
        sigma2 = var_y.copy()
        odd_max = (N - 1) // 2

        # linear cross term
        cross = np.zeros_like(x, dtype=float)
        for m in range(0, odd_max + 1):
            Pm = self.poly_derivative(x, coeffs, m=2 * m + 1)
            cross += Pm / (2.0**m * factorial(m)) * var_x**m
        sigma2 += -2.0 * cov_xy * cross

        # odd-odd double sum
        oo = np.zeros_like(x, dtype=float)
        for m in range(0, odd_max + 1):
            Pm = self.poly_derivative(x, coeffs, m=2 * m + 1)
            for l in range(0, odd_max + 1):
                Pl = self.poly_derivative(x, coeffs, m=2 * l + 1)
                coef = comb(2 * m + 2 * l + 2, 2 * m + 1) / (
                    2.0 ** (m + l + 1) * factorial(m + l + 1)
                )
                oo += coef * Pm * Pl * var_x ** (m + l + 1)
        sigma2 += oo

        # even-even double sum
        even_max = N // 2
        ee = np.zeros_like(x, dtype=float)
        for m in range(1, even_max + 1):
            Pm = self.poly_derivative(x, coeffs, m=2 * m)
            for l in range(1, even_max + 1):
                Pl = self.poly_derivative(x, coeffs, m=2 * l)
                coef = (comb(2 * m + 2 * l, 2 * m) - comb(m + l, m)) / (
                    2.0 ** (m + l) * factorial(m + l)
                )
                ee += coef * Pm * Pl * var_x ** (m + l)
        sigma2 += ee

        return Delta, sigma2

    def ln_likelihood(self, pars):
        coeffs = pars[:self.order + 1]

        sigma_x_int = pars[self.order + 1]
        sigma_y_int = pars[self.order + 2]
        sigma_xy_int = pars[self.order + 3]

        var_x = self.x_err**2 + sigma_x_int**2
        var_y = self.y_err**2 + sigma_y_int**2
        cov_xy = self.xy_err + sigma_xy_int

        P = self.poly_model(self.x, coeffs)
        observed_residual = self.y - P
        Delta, variance = self._effective_variance(self.x, coeffs, var_x, var_y, cov_xy, observed_residual)

        if np.any(~np.isfinite(variance)):
            return -np.inf

        if np.any(variance <= 0):
            return -np.inf

        return -0.5 * np.sum(np.log(2.0 * np.pi * variance) + Delta**2 / variance)

    def ln_prior(self, pars):
        coeffs = pars[:self.order + 1]
        sigma_x_int = pars[self.order + 1]
        sigma_y_int = pars[self.order + 2]
        sigma_xy_int = pars[self.order + 3]

        if np.any(np.abs(coeffs) > 1e6):
            return -np.inf

        if sigma_x_int < 0 or sigma_x_int > 1e6:
            return -np.inf

        if sigma_y_int < 0 or sigma_y_int > 1e6:
            return -np.inf

        if abs(sigma_xy_int) > sigma_x_int * sigma_y_int:
            return -np.inf

        return 0

    def ln_posterior(self, pars):
        lp = self.ln_prior(pars)

        if not np.isfinite(lp):
            return -np.inf

        ll = self.ln_likelihood(pars)

        if not np.isfinite(ll):
            return -np.inf

        return lp + ll

    def __call__(self, pars):
        return self.ln_posterior(pars)


def hoggfit(x, y, xerr=None, yerr=None, xyerr=None, order=1, nwalkers=32, nsteps=10000, ncores=None):
    rng = np.random.default_rng()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xerr = np.zeros_like(x) if xerr is None else np.asarray(xerr, dtype=float)
    yerr = np.zeros_like(y) if yerr is None else np.asarray(yerr, dtype=float)
    xyerr = np.zeros_like(x) if xyerr is None else np.asarray(xyerr, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(xerr) & np.isfinite(yerr) & np.isfinite(xyerr)

    x = x[mask]
    y = y[mask]
    xerr = xerr[mask]
    yerr = yerr[mask]
    xyerr = xyerr[mask]

    idx = np.argsort(x)

    x = x[idx]
    y = y[idx]
    xerr = xerr[idx]
    yerr = yerr[idx]
    xyerr = xyerr[idx]

    model = PolynomialModel(x, y, yerr, xerr, xyerr, order=order)
    ndim = model.ndim

    coeffs0 = npoly.polyfit(x, y, order)
    y0 = model.poly_model(x, coeffs0)
    residual = y - y0
    sigma0 = max(np.std(residual), 1e-6)

    sigma_x_int0 = max(0.01 * np.std(x), 1e-6)
    sigma_y_int0 = sigma0
    sigma_xy_int0 = 0.0

    print("\n------------------------------------------")
    print("Initial fit diagnostics")
    print("------------------------------------------")
    print("Polynomial coefficients:", coeffs0)
    print("Initial residual scatter:", sigma0)
    print("Initial RMSE:", np.sqrt(np.mean(residual**2)))

    initial_params_test = np.concatenate([coeffs0, [sigma_x_int0, sigma_y_int0, sigma_xy_int0]])
    print("Initial log posterior:", model.ln_posterior(initial_params_test))
    print("------------------------------------------\n")
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

    initial_params = np.zeros((nwalkers, ndim))

    for k in range(order + 1):
        scale = max(abs(coeffs0[k]) * 1e-3, 1e-6)
        initial_params[:, k] = coeffs0[k] + scale * rng.standard_normal(nwalkers)

    initial_params[:, order + 1] = np.abs(sigma_x_int0 * (1.0 + 0.05 * rng.standard_normal(nwalkers)))
    initial_params[:, order + 2] = np.abs(sigma_y_int0 * (1.0 + 0.05 * rng.standard_normal(nwalkers)))

    sx = initial_params[:, order + 1]
    sy = initial_params[:, order + 2]
    max_covariance = sx * sy

    initial_params[:, order + 3] = 0.1 * max_covariance * rng.standard_normal(nwalkers)
    initial_params[:, order + 3] = np.clip(initial_params[:, order + 3], -0.99 * max_covariance, 0.99 * max_covariance)

    if ncores is None:
        ncores = mp.cpu_count()

    with mp.Pool(processes=ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, model.ln_posterior, pool=pool)
        sampler.run_mcmc(initial_params, nsteps, progress=True)

    return sampler


def analyze_hoggfit(sampler, x, y, order=1, intrinsic_true=None, truths=None, burnin=250, thin=10, nsamp=200):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    chain = sampler.get_chain()
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    poly_samples = samples[:, :order + 1]
    sigma_x_samples = samples[:, order + 1]
    sigma_y_samples = samples[:, order + 2]
    sigma_xy_samples = samples[:, order + 3]

    full_samples = np.column_stack((poly_samples, sigma_x_samples, sigma_y_samples, sigma_xy_samples))

    labels = [rf"$a_{{{i}}}$" for i in range(order, -1, -1)]
    labels += [r"$\sigma_{\rm int,x}$", r"$\sigma_{\rm int,y}$", r"$\sigma_{\rm int,xy}$"]

    fig, axes = plt.subplots(full_samples.shape[1], 1, figsize=(10, 2.2 * full_samples.shape[1]), sharex=True)

    if full_samples.shape[1] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(chain[:, :, i], color="black", alpha=0.25, lw=0.5)
        ax.set_ylabel(labels[::-1][i])

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()

    truths_plot = None

    if truths is not None:
        truths_plot = list(truths)

        if intrinsic_true is not None:
            truths_plot.extend(list(intrinsic_true))

    corner_labels = [rf"$a_{{{i}}}$" for i in range(order + 1)]
    corner_labels += [r"$\sigma_{\rm int,x}$", r"$\sigma_{\rm int,y}$", r"$\sigma_{\rm int,xy}$"]

    fig = corner.corner(
        samples,
        labels=corner_labels,
        truths=truths_plot,
        show_titles=True,
        title_fmt=".3g",
        quantiles=[0.16, 0.50, 0.84],
        fill_contours=True,
        smooth=1.0,
        smooth1d=1.0,
        levels=(0.393, 0.865, 0.989),
        plot_datapoints=True
    )

    fig.set_size_inches(9, 9)
    plt.show()

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

    q16, q50, q84 = np.percentile(full_samples, [16, 50, 84], axis=0)

    print("\n------------------------------------------")
    print("Posterior Polynomial Coefficients")
    print("------------------------------------------")

    for power in range(order, -1, -1):
        i = power
        print(f"a_{power:<2d} = {q50[i]:.6e} (+{q84[i] - q50[i]:.6e}, -{q50[i] - q16[i]:.6e})")

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

    covariance_matrix = np.cov(full_samples, rowvar=False)

    print("\n------------------------------------------")
    print("Posterior Parameter Covariance Matrix")
    print("------------------------------------------")
    print(covariance_matrix)

    return [q16, q50, q84], covariance_matrix, intrinsic_covariance