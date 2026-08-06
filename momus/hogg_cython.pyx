import numpy as np
cimport numpy as np
from libc.math cimport log, pi


def ln_likelihood(
    np.ndarray[np.float64_t, ndim=1] coeffs,
    double sigma_int,
    np.ndarray[np.float64_t, ndim=1] x,
    np.ndarray[np.float64_t, ndim=1] y,
    np.ndarray[np.float64_t, ndim=1] y_err,
    int order
):

    cdef int n = x.shape[0]
    cdef int i
    cdef double model
    cdef double residual
    cdef double variance
    cdef double loglike = 0.0

    for i in range(n):

        # polynomial evaluation (Horner's method)
        model = coeffs[0]

        for j in range(1, order+1):
            model = model*x[i] + coeffs[j]


        residual = y[i] - model

        variance = y_err[i]*y_err[i] + sigma_int*sigma_int


        if variance <= 0:
            return -np.inf


        loglike += (
            residual*residual/variance
            +
            log(2*pi*variance)
        )


    return -0.5*loglike