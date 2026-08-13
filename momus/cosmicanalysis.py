#-----------------------------------------------------------------------#
# momus.cosmicanalysis v0.1.0
# By Hunter Brooks, at UToledo
#-----------------------------------------------------------------------#

# Import Packages
#-----------------------------------------------------------------------#
import matplotlib.pyplot as plt
import numpy as np
import corner
#-----------------------------------------------------------------------#

# Import Other Files
#-----------------------------------------------------------------------#
from momus.latexreader import read_latex_equation
#-----------------------------------------------------------------------#



# Plot Walker Steps
#-----------------------------------------------------------------------#
def stepplot(full_samples, chain, labels):

    # Setup Empty Figure
    fig, axes = plt.subplots(full_samples.shape[1], 1, figsize=(10, 2.2 * full_samples.shape[1]), sharex=True)
    
    # Make Axes Iterable
    if full_samples.shape[1] == 1:
        axes = [axes]

    # Plot Walker Chains
    for i, ax in enumerate(axes):
        ax.plot(chain[:, :, i], color="black", alpha=0.25, lw=0.5)
        ax.set_ylabel(labels[i])

    # Make Plot Pretty
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()
    return
#-----------------------------------------------------------------------#



# Plot Fitted Relation with Associated Confidence Interval
#-----------------------------------------------------------------------#
def normalplot(samples, x, y, model_func, num_coeffs, nsamp):

    # Generate Random Posterior Samples
    rng = np.random.default_rng()
    inds = rng.choice(len(samples), size=min(nsamp, len(samples)), replace=False)

    # Generate x Values
    x_dense = np.linspace(x.min(), x.max(), 1000)

    # Evaluate Posterior Models
    curves = []
    for i in inds:
        coeffs = samples[i, :num_coeffs]
        curve = model_func(x_dense, *coeffs)
        curves.append(np.asarray(curve, dtype=float))
    curves = np.asarray(curves)

    # Calculate Confidence Interval
    lower, upper = np.percentile(curves, [16, 84], axis=0)
    median_curve = np.median(curves, axis=0)

    # Create Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, color="black", zorder=1)
    ax.fill_between(x_dense, lower, upper, color="orangered", alpha=0.3, label=r"1$\sigma$ CI")
    ax.plot(x_dense, median_curve, color="orangered", lw=2, label="Median model")

    # Make Plot Pretty
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid(alpha=0.15)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return
#-----------------------------------------------------------------------#



# Corner Plot
#-----------------------------------------------------------------------#
def cornerplot(truths, intrinsic_true, labels, samples):

    # Check Whether Truths Are Provided
    truths_plot = None
    if truths is not None:
        truths_plot = list(truths)
        if intrinsic_true is not None:
            truths_plot.extend(list(intrinsic_true))

    # Create Corner Plot
    fig = corner.corner(samples, labels=labels, truths=truths_plot, show_titles=True, title_fmt=".3g", quantiles=[0.16, 0.50, 0.84], fill_contours=True, smooth=1.0, smooth1d=1.0, levels=(0.393, 0.865, 0.989), plot_datapoints=True)
    fig.set_size_inches(9, 9)
    plt.show()
    return
#-----------------------------------------------------------------------#



# Plot Correlation Matrix
#-----------------------------------------------------------------------#
def correlationplot(samples, labels):

    # Create Correlation Matrix
    corr = np.corrcoef(samples, rowvar=False)

    # Create Figure
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)

    # Set Labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Make Plot Pretty
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Posterior Correlation Matrix")
    plt.tight_layout()
    plt.show()
    return
#-----------------------------------------------------------------------#



# Simple Analysis of Outputted MCMC Sampler
#-----------------------------------------------------------------------#
def CosmicAnalysis(sampler, x, y, latex_eq=r"a_0 + a_1 x", intrinsic_true=None, truths=None, burnin=250, thin=10, nsamp=200):

    # Obtain Observed x and y Values
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Read LaTeX Equation
    (expr, x_symbol, coefficients, num_coeffs, model_func) = read_latex_equation(latex_eq)

    # Create Parameter Labels
    coefficient_labels = [rf"$a_{{{i}}}$" for i in range(num_coeffs)]
    scatter_labels = [r"$\sigma_{\rm int,x}$", r"$\sigma_{\rm int,y}$", r"$\sigma_{\rm int,xy}$"]
    labels = (coefficient_labels + scatter_labels)

    # Flatten Sampler
    chain = sampler.get_chain()
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    # Check Sample Dimensions
    expected_ndim = num_coeffs + 3
    if samples.shape[1] != expected_ndim:
        raise ValueError(
            f"Sampler contains {samples.shape[1]} parameters, "
            f"but equation requires {expected_ndim} "
            f"({num_coeffs} coefficients + 3 intrinsic scatter parameters)."
        )

    # Separate Coefficients
    coeff_samples = samples[:, :num_coeffs]

    # Separate Intrinsic Scatter
    sigma_x_samples = samples[:, num_coeffs]
    sigma_y_samples = samples[:, num_coeffs + 1]
    sigma_xy_samples = samples[:, num_coeffs + 2]

    # Plot Sampler
    stepplot(samples, chain, labels)
    cornerplot(truths, intrinsic_true, labels, samples)
    normalplot(samples, x, y, model_func, num_coeffs, nsamp)
    correlationplot(samples, labels)


    # Calculate Posterior Quantiles
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)

    # Print Coefficients
    print("\n------------------------------------------")
    print("Posterior Coefficients")
    print("------------------------------------------")
    for i in range(num_coeffs):
        print(
            f"a_{i:<2d} = "
            f"{q50[i]:.6e} "
            f"(+{q84[i] - q50[i]:.6e}, "
            f"-{q50[i] - q16[i]:.6e})"
        )

    # Print Cosmic Scatter
    print("\n------------------------------------------")
    print("Cosmic Scatter")
    print("------------------------------------------")
    sx_q16, sx_q50, sx_q84 = np.percentile(sigma_x_samples, [16, 50, 84])
    sy_q16, sy_q50, sy_q84 = np.percentile(sigma_y_samples, [16, 50, 84])
    sxy_q16, sxy_q50, sxy_q84 = np.percentile(sigma_xy_samples, [16, 50, 84])
    print(
        f"sigma_int_x  = "
        f"{sx_q50:.6e} "
        f"(+{sx_q84 - sx_q50:.6e}, "
        f"-{sx_q50 - sx_q16:.6e})"
    )
    print(
        f"sigma_int_y  = "
        f"{sy_q50:.6e} "
        f"(+{sy_q84 - sy_q50:.6e}, "
        f"-{sy_q50 - sy_q16:.6e})"
    )
    print(
        f"sigma_int_xy = "
        f"{sxy_q50:.6e} "
        f"(+{sxy_q84 - sxy_q50:.6e}, "
        f"-{sxy_q50 - sxy_q16:.6e})"
    )


    # Print Covariance Matrix
    print("\n------------------------------------------")
    print("Covariance Matrix")
    print("------------------------------------------")
    covariance_matrix = np.cov(samples, rowvar=False)
    print(covariance_matrix)
    return ([q16, q50, q84], covariance_matrix)
#-----------------------------------------------------------------------#