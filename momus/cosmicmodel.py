#-----------------------------------------------------------------------#
# momus.cosmicmodel v0.2.0 (Generalized LaTeX Version)
# By Hunter Brooks, at UToledo, Toledo
#-----------------------------------------------------------------------#

# Import Packages
#-----------------------------------------------------------------------#
from numpy.polynomial.legendre import leggauss
import numpy as np
#-----------------------------------------------------------------------#

# Import Other momus Files
#-----------------------------------------------------------------------#
from momus.latexreader import read_latex_equation
#-----------------------------------------------------------------------#



# Define Class for General Relation with Cosmic Scatter
#-----------------------------------------------------------------------#
class CosmicModel:

    # Initialize Model
    #-------------------------------------------------------------------#
    def __init__(self, x, y, y_err, x_err, xy_err, latex_eq=r"a_0 + a_1 x", quad_points=67):

        # Ensures Variables are Correctly Assigned
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.x_err = np.asarray(x_err, dtype=float)
        self.y_err = np.asarray(y_err, dtype=float)
        self.xy_err = np.asarray(xy_err, dtype=float)

        # Read LaTeX Equation
        self.latex_eq = latex_eq
        self.expr, self.x_symbol, self.coefficients, self.num_coeffs, self.model_func = read_latex_equation(latex_eq)

        # Total Number of Parameters
        self.ndim = self.num_coeffs + 3

        # Set Number of Quadrature Points
        self.quad_points = quad_points
        self.gl_nodes, self.gl_weights = leggauss(quad_points)
        self.log_2pi = np.log(2.0 * np.pi)
    #-------------------------------------------------------------------#


    # Define Custom Evaluated Model
    #-------------------------------------------------------------------#
    def evaluate_model(self, x, coeffs):
        return self.model_func(x, *coeffs)
    #-------------------------------------------------------------------#


    # Define Log Likelihood for Generative Model
    #-------------------------------------------------------------------#
    def ln_likelihood(self, pars):

        # Separate LaTeX Coefficients and Intrinsic Scatter
        coeffs = pars[:self.num_coeffs]
        sigma_x_int = pars[self.num_coeffs]
        sigma_y_int = pars[self.num_coeffs + 1]
        sigma_xy_int = pars[self.num_coeffs + 2]

        # Calculate Total Covariance
        var_x = self.x_err**2 + sigma_x_int**2
        var_y = self.y_err**2 + sigma_y_int**2
        cov_xy = self.xy_err + sigma_xy_int

        # Calculate Determinant of Total Covariance Matrix
        det = var_x * var_y - cov_xy**2
        if np.any(~np.isfinite(det)) or np.any(det <= 0.0):
            return -np.inf

        # Define Integration Limits for Latent True x
        sigma_x_total = np.sqrt(var_x)
        lower = self.x - 10.0 * sigma_x_total
        upper = self.x + 10.0 * sigma_x_total

        # Transform Gauss-Legendre Nodes to Integration Interval
        xi = (0.5 * (upper[:, None] - lower[:, None]) * self.gl_nodes[None, :] + 0.5 * (upper[:, None] + lower[:, None]))
        weights = (0.5 * (upper[:, None] - lower[:, None]) * self.gl_weights[None, :])

        # Evaluate Model at Every Latent x Value and Residuals and Total Covariance Matrix
        P_xi = self.evaluate_model(xi, coeffs)
        dx = self.x[:, None] - xi
        dy = self.y[:, None] - P_xi
        inv00 = var_y / det
        inv01 = -cov_xy / det
        inv11 = var_x / det

        # Calculate Likelihood 
        chi2 = (inv00[:, None] * dx**2 + 2.0 * inv01[:, None] * dx * dy + inv11[:, None] * dy**2)
        logp = (-self.log_2pi - 0.5 * np.log(det)[:, None] - 0.5 * chi2)
        integrand = np.exp(logp) * weights
        integral = np.sum(integrand, axis=1)

        # Calculate Total Log Likelihood
        if np.any(~np.isfinite(integral)) or np.any(integral <= 0.0):
            return -np.inf
        return np.sum(np.log(integral))
    #-------------------------------------------------------------------#


    # Define Prior Distribution
    #-------------------------------------------------------------------#
    def ln_prior(self, pars):

        # Separate LaTeX Coefficients and Intrinsic Scatter
        coeffs = pars[:self.num_coeffs]
        sigma_x_int = pars[self.num_coeffs]
        sigma_y_int = pars[self.num_coeffs + 1]
        sigma_xy_int = pars[self.num_coeffs + 2]

        # Reject Bad Values
        if not np.all(np.isfinite(pars)):
            return -np.inf
        if np.any(np.abs(coeffs) > 1e6):
            return -np.inf
        if sigma_x_int <= 0.0 or sigma_x_int > 1e6:
            return -np.inf
        if sigma_y_int <= 0.0 or sigma_y_int > 1e6:
            return -np.inf
        if abs(sigma_xy_int) > sigma_x_int * sigma_y_int:
            return -np.inf
        return 0.0
    #-------------------------------------------------------------------#


    # Define Log Posterior Probability
    #-------------------------------------------------------------------#
    def ln_posterior(self, pars):
        lp = self.ln_prior(pars)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.ln_likelihood(pars)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll
    #-------------------------------------------------------------------#


    # Define Callable Model
    #-------------------------------------------------------------------#
    def __call__(self, pars):
        return self.ln_posterior(pars)
    #-------------------------------------------------------------------#
#-----------------------------------------------------------------------#