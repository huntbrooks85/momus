#-----------------------------------------------------------------------#
# momus.cosmicfit v0.1.0
# By Hunter Brooks, at UToledo
#-----------------------------------------------------------------------#

# Import Packages
#-----------------------------------------------------------------------#
from multiprocessing.pool import ThreadPool
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import emcee
#-----------------------------------------------------------------------#

# Import Other momus Files
#-----------------------------------------------------------------------#
from momus.cosmicmodel import CosmicModel
#-----------------------------------------------------------------------#



# Define Function to run MCMC to Fit Relation with Cosmic Scatter
#-----------------------------------------------------------------------#
def CosmicFit(x, y, xerr=None, yerr=None, xyerr=None, nwalkers=32, nsteps=5000, ncores=None, quad_points=67, latex_eq=r"a_0 + a_1 x"):

    # Ensures Variables are Correctly Assigned
    #-------------------------------------------------------------------#
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xerr = np.zeros_like(x) if xerr is None else np.asarray(xerr, dtype=float)
    yerr = np.zeros_like(y) if yerr is None else np.asarray(yerr, dtype=float)
    xyerr = np.zeros_like(x) if xyerr is None else np.asarray(xyerr, dtype=float)
    #-------------------------------------------------------------------#


    # Map Out Bad Variable Values and Sort Them
    #-------------------------------------------------------------------#
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(xerr) & np.isfinite(yerr) & np.isfinite(xyerr)

    x, y = x[mask], y[mask]
    xerr, yerr = xerr[mask], yerr[mask]
    xyerr = xyerr[mask]

    idx = np.argsort(x)

    x, y = x[idx], y[idx]
    xerr, yerr = xerr[idx], yerr[idx]
    xyerr = xyerr[idx]
    #-------------------------------------------------------------------#


    # Load in Model
    #-------------------------------------------------------------------#
    model = CosmicModel(x, y, yerr, xerr, xyerr, latex_eq=latex_eq, quad_points=quad_points)

    ncoeffs = model.num_coeffs
    ndim = model.ndim
    #-------------------------------------------------------------------#


    # Print Model Information
    #-------------------------------------------------------------------#
    print("\n------------------------------------------")
    print("CosmicFit Model")
    print("------------------------------------------")
    print("LaTeX equation:", latex_eq)
    print("Coefficients:", model.coefficients)
    print("Number of coefficients:", ncoeffs)
    print("Total parameters:", ndim)
    print("------------------------------------------\n")
    #-------------------------------------------------------------------#


    # Fit Initial Coefficients
    #-------------------------------------------------------------------#
    def residual_function(coeffs):
        y_model = model.evaluate_model(x, coeffs)
        if not np.all(np.isfinite(y_model)):
            return np.full_like(y, 1e100)
        return y - y_model

    coeffs_guess = np.ones(ncoeffs)
    result = least_squares(residual_function, coeffs_guess)

    if result.success:
        coeffs0 = result.x
    else:
        print("WARNING: Initial coefficient fit did not converge.")
        coeffs0 = coeffs_guess
    #-------------------------------------------------------------------#


    # Initial Cosmic Scatter
    #-------------------------------------------------------------------#
    y0 = model.evaluate_model(x, coeffs0)

    residual = y - y0

    sigma0 = max(np.std(residual), 1e-6)
    sigma_x_int0 = max(0.01 * np.std(x), 1e-6)
    sigma_y_int0 = sigma0
    sigma_xy_int0 = 0.0
    #-------------------------------------------------------------------#


    # Print Out Initial Conditions
    #-------------------------------------------------------------------#
    print("\n------------------------------------------")
    print("Initial fit diagnostics")
    print("------------------------------------------")
    print("Initial coefficients:", coeffs0)
    print("Initial residual scatter:", sigma0)
    print("Initial RMSE:", np.sqrt(np.mean(residual**2)))
    initial_params_test = np.concatenate([coeffs0, [sigma_x_int0, sigma_y_int0, sigma_xy_int0]])
    print("Initial log posterior:", model.ln_posterior(initial_params_test))
    print("------------------------------------------\n")
    #-------------------------------------------------------------------#


    # Plot Initial Conditions
    #-------------------------------------------------------------------#
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="black", label="Data")
    x_plot = np.linspace(x.min(), x.max(), 1000)
    y_plot = model.evaluate_model(x_plot, coeffs0)
    plt.plot(x_plot, y_plot, color="red", lw=2, label="Initial model")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
    #-------------------------------------------------------------------#


    # Set Up Walkers Around Initial Conditions
    #-------------------------------------------------------------------#
    initial_params = np.zeros((nwalkers, ndim))
    rng = np.random.default_rng()
    #-------------------------------------------------------------------#


    # Small Variations in Coefficients
    #-------------------------------------------------------------------#
    for k in range(ncoeffs):
        scale = max(abs(coeffs0[k]) * 1e-3, 1e-6)
        initial_params[:, k] = (coeffs0[k] + scale * rng.standard_normal(nwalkers))
    #-------------------------------------------------------------------#


    # 5% Variation For Intrinsic x Scatter
    #-------------------------------------------------------------------#
    initial_params[:, ncoeffs] = np.abs(sigma_x_int0 * (1.0 + 0.05 * rng.standard_normal(nwalkers)))
    initial_params[:, ncoeffs] = np.maximum(initial_params[:, ncoeffs], 1e-8)
    #-------------------------------------------------------------------#


    # 5% Variation For Intrinsic y Scatter
    #-------------------------------------------------------------------#
    initial_params[:, ncoeffs + 1] = np.abs(sigma_y_int0 * (1.0 + 0.05 * rng.standard_normal(nwalkers)))
    initial_params[:, ncoeffs + 1] = np.maximum(initial_params[:, ncoeffs + 1], 1e-8)
    #-------------------------------------------------------------------#


    # Intrinsic xy Covariance
    #-------------------------------------------------------------------#
    sx = initial_params[:, ncoeffs]
    sy = initial_params[:, ncoeffs + 1]

    max_covariance = sx * sy
    initial_params[:, ncoeffs + 2] = (0.1 * max_covariance * rng.standard_normal(nwalkers))
    initial_params[:, ncoeffs + 2] = np.clip(initial_params[:, ncoeffs + 2], -0.99 * max_covariance, 0.99 * max_covariance)
    #-------------------------------------------------------------------#


    # Start MCMC Simulation
    #-------------------------------------------------------------------#
    if ncores is None:
        ncores = mp.cpu_count()

    moves = [
            (emcee.moves.DESnookerMove(), 0.05),
            (emcee.moves.DEMove(gamma0=1.2), 0.4),
            (emcee.moves.StretchMove(a=5), 0.55)
            ] 
    with ThreadPool(processes=ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, model.ln_posterior, pool=pool, moves=moves)
        sampler.run_mcmc(initial_params, nsteps, progress=True)
    #-------------------------------------------------------------------#


    # Return Sampler
    #-------------------------------------------------------------------#
    return sampler
    #-------------------------------------------------------------------#
#-----------------------------------------------------------------------#