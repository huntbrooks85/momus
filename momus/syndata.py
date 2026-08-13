#-----------------------------------------------------------------------#
# momus.syndata v0.1.0
# By Hunter Brooks, at UToledo
#-----------------------------------------------------------------------#

# Import Packages
#-----------------------------------------------------------------------#
import numpy as np
#-----------------------------------------------------------------------#


# Import Other momus Files
#-----------------------------------------------------------------------#
from momus.latexreader import read_latex_equation
#-----------------------------------------------------------------------#


# Synthetic Data Generator
#-----------------------------------------------------------------------#
def generate_synthetic_data(n_data, latex_eq, coeffs_true, x_range=(-2, 5), sigma_x_int=0.1, sigma_y_int=0.25, sigma_xy_int=0.0, x_err_max=0.0, y_err_max=0.0, seed=1):

    # Set Seed for Reproducibility
    #-------------------------------------------------------------------#
    np.random.seed(seed)
    #-------------------------------------------------------------------#


    # Read LaTeX Equation
    #-------------------------------------------------------------------#
    expr, x_symbol, coefficients, num_coeffs, model_func, domain = read_latex_equation(latex_eq)
    if domain[0] == False: 
        raise ValueError(
            f"Equation is not continous over all real values."
            f"Valid range for inputted equation: {domain[1]}"
        )
    #-------------------------------------------------------------------#


    # Check Coefficient Count
    #-------------------------------------------------------------------#
    if len(coeffs_true) != num_coeffs:
        raise ValueError(
            f"Equation requires {num_coeffs} coefficients "
            f"({coefficients}), but {len(coeffs_true)} were supplied."
        )
    #-------------------------------------------------------------------#


    # Generate True Values
    #-------------------------------------------------------------------#
    x_true = np.sort(np.random.uniform(x_range[0], x_range[1], n_data))
    y_true = model_func(x_true, *coeffs_true)
    y_true = np.asarray(y_true, dtype=float)
    #-------------------------------------------------------------------#


    # Measurement Errors
    #-------------------------------------------------------------------#
    x_err = np.random.uniform(0, x_err_max, n_data)
    y_err = np.random.uniform(0, y_err_max, n_data)

    x_noise = np.random.normal(0, x_err)
    y_noise = np.random.normal(0, y_err)
    #-------------------------------------------------------------------#


    # Intrinsic Scatter
    #-------------------------------------------------------------------#
    intrinsic_cov = np.array([[sigma_x_int**2, sigma_xy_int], [sigma_xy_int, sigma_y_int**2]])
    if np.linalg.det(intrinsic_cov) <= 0:
        raise ValueError(
            "Intrinsic covariance matrix must be positive definite. "
            "Require |sigma_xy_int| < sigma_x_int * sigma_y_int."
        )
    intrinsic_noise = np.random.multivariate_normal(mean=[0.0, 0.0], cov=intrinsic_cov, size=n_data)
    x_int = intrinsic_noise[:, 0]
    y_int = intrinsic_noise[:, 1]
    #-------------------------------------------------------------------#


    # Generate Observed Values
    #-------------------------------------------------------------------#
    x_obs = x_true + x_noise + x_int
    y_obs = y_true + y_noise + y_int
    #-------------------------------------------------------------------#


    # Return Data
    #-------------------------------------------------------------------#
    return x_true, y_true, x_obs, y_obs, x_err, y_err
    #-------------------------------------------------------------------#
#-----------------------------------------------------------------------#