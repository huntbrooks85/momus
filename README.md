<p align="center">
    <a href="https://ibb.co/v6CbwTzT"><img src="/misc/Momus Logo.png" width="40%"></a> <br>
</p>

<div align="center">
  <p id="description" > <b>momus</b> is a Python package for regression fitting using a Markov-Chain Monte-Carlo simulation to fit for desired coefficients and intrinsic scatter. The desired function must be continuous with any number of fitted coefficients.  </p>
</div>

<div align="center">
  <h2 style="font-size: 2em;"> 🔍 Example 🔎 </h2>
</div>

<div align="center">
  <p style="font-size: 1.2em;"> <b>
      There is an example notebook provided on the GitHub repository. 
    </b> </p>
</div>

<div align="center">
  <h2 style="font-size: 2em;">🛠️ Installation 🛠️</h2>
</div>

<div align="center">
<pp><b> pip Installation </b><pp>
</div>
<div align="center">
</div>

1. **Download Python:** Visit [here](https://www.python.org/downloads/) to install Python 
2. **Download pip:** Visit [here](https://pip.pypa.io/en/stable/installation/) to install pip
3. **Run Install Command:** Run the command in terminal:
```bash
   pip install momus
```

<div align="center">
  <h2 style="font-size: 2em;">🤔 General Logic 🤔</h2>
</div>
<p align="center">
    <a href="https://ibb.co/kgGdPYfg"><img src="/misc/Momus Diagram.drawio.pdf" width="40%"></a> <br>
</p>

<div align="center">
  <h2 style="font-size: 2em;">🧮 Using momus 🧮</h2>
</div>

### What are The Valid Functions?
Any function that is continuous over -$\infty$ to $\infty$. This will be tested and if not valid a ```ValueError``` will be returned. To write the equations we provide easy access using LaTeX format. The dependent variable should always be written as x and all coefficients should be written as $\{a_0, a_1, ..., a_{n-1}, a_n \}$. To test your LaTeX equation refer to [LaTeX Equation Editor](https://latexeditor.lagrida.com/).

A few examples are shown below: 

1. $a_0 + a_1 x + a_2x^2 + a_3 x^3 + a_4 x^4$: ```r"a_0 + a_1 x + a_2x^2 + a_3 x^3 + a_4 x^4"```
2. $a_0 \sin(a_1 x)$: ```r"a_0 \sin(a_1 x)"```
3. $a_0 e^{a_1 x}$: ```a_0 e^{a_1 x}```
4. $a_0 \cos(\pi x)$: ```a_0 \cos(\pi x)```

### How to Use <code> generate_synthetic_data </code>

To test whether the package is working and the function you want to use is highly divergent then please use ```generate_synthetic_data```. This creates synthetic data with known coefficients, observed uncertainties, and intrinsic scatter values to test against. 

After installing momus import the synthetic data function:  ```from momus.syndata import generate_synthetic_data```. From which, define your variables as listed below to plug into the function and execute ```generate_synthetic_data(variables)```. Returns the true x and y values, the observed x and y values, and the observed x and y uncertainties (all as lists).

All variables for ```generate_synthetic_data``` are: 
- **Required Variables:**
  - **n_data:** Number of observed data points: *int*:
     - *example:* ```250```
  - **latex_eq:** LaTeX equation: *string*:
     - *example:* ```r"a_0 \sin(a_1 x)"```
  - **coeffs_true:** The values of the true coefficient values: *list*:
     - *example:* ```[2.5, 0.9]```

- **User Configuration:**
  - **x_range:** The observed x-range to sample from: *tuple*
    - default=```(-2, 5)```
  - **sigma_x_int:** Intrinsic scatter in x: *float*
    - default=```0.1```
  - **sigma_y_int:** Intrinsic scatter in y: *float*
    - default=```0.25```
  - **sigma_xy_int:** Intrinsic covariance between x and y: *float*
    - default=```0.0```
  - **x_err_max:** Max observed x uncertainty: *float*
    - default=```0.0```
  - **y_err_max:** Max observed y uncertainty: *float*
    - default=```0.0```
  - **seed:** Numpy random seed: *int*
    - default=```1```

### How to Use <code> CosmicFit </code>
This is the main function of our package. It performs a Markov-Chain Monte-Carlo (MCMC) simulation with numerical integration using Gauss-Legendre Quadrature moments to stably integrate over $p(x, y|\Theta, C_{\rm int})$.

After observed data is assigned, import the ```CosmicFit``` package using: ```from momus.cosmicfit import CosmicFit```. From this define all desired variables and execute ```CosmicFit(variables)```. Returns the total ```emcee``` MCMC sampler (```numpy``` array). 

All variables for ```CosmicFit``` are: 
- **Required Variables:**
  - **x:** Observed x list: *list*:
     - *example:* ```[1, 2, 3, 4]```
  - **y:** Observed y list: *list*:
     - *example:* ```[2, 4, 5, 7]```

- **User Configuration:**
  - **xerr:** The observed x uncertainties: *list*
    - default=```None```
  - **yerr:** The observed y uncertainties: *list*
    - default=```None```
  - **xyerr:** The observed x-y covariance: *list*
    - default=```None```
  - **nwalkers:** Number of walkers used in MCMC simulation: *int*
    - default=```32```
  - **nsteps:** Number of steps used in MCMC simulation: *int*
    - default=```5000```
  - **ncores:** Number of cores used in multithreading: *int*
    - default=```None```
  - **quad_points:** Number of quadrupole moments: *int*
    - default=```67```
  - **latex_eq:** Desired LaTeX equation: *string*
    - default=```r"a_0 + a_1 x"```

### How to Use <code> CosmicAnalysis </code>
After the MCMC simulation is finished a quick analysis of the data can be done with ```CosmicAnalysis```. Which returns the best fit coefficients, intrinsic scatter, the relevant uncertainties, and the covariance matrix values alongside plots of the walkers in each parameter, corner plot, the best fit line with $1\sigma$ confidence intervals, and a correlation matrix figure. 

Import the ```CosmicAnalysis``` package using: ```from momus.cosmicanalysis import CosmicAnalysis```. From this define all desired variables and execute ```CosmicAnalysis(variables)```.

All variables for ```CosmicAnalysis``` are: 
- **Required Variables:**
  - **sampler:** The ```emcee``` sampler: *list*:
     - *example:* ```[1, 2, 3, 4]```
  - **x:** Observed x list: *list*:
     - *example:* ```[1, 2, 3, 4]```
  - **y:** Observed y list: *list*:
     - *example:* ```[2, 4, 5, 7]```

- **User Configuration:**
  - **latex_eq:** Desired LaTeX equation: *string*
    - default=```r"a_0 + a_1 x"```
  - **intrinsic_true:** The list of true intrinsic values if available (```[sigma_x_int_true, sigma_y_int_true, sigma_xy_int_true]```): *list*
    - default=```None```
  - **truths:** The list of true coefficient values if available: *list*
    - default=```None```
  - **burnin:** Number of discarded steps: *int*
    - default=```250```
  - **thin:** Only keep the nth step in the walker path: *int*
    - default=```10```
  - **nsamp:** Number of samples shown in figure, to save on storage: *int*
    - default=```200```

<div align="center">
  <h2 style="font-size: 2em;">📞 Support Team 📞</h2>
</div>

- **Mr. Hunter Brooks**
  - hbrooks8 (at) rockets.utoledo.edu
- **Dr. Michael Cushing**

<div align="center">
  <h2 style="font-size: 2em;">📖 Acknowledgments 📖</h2>
</div>

If you intend to publish any calculations done by momus, please reference Brooks et al. (in prep).