# ONLY SIGMA Y
# ==================================================================== #
class PolynomialModel(object):

    def __init__(self,x,y,y_err,x_err,xy_err,order=1):
        self.x=np.asarray(x,dtype=float)
        self.y=np.asarray(y,dtype=float)
        self.y_err=np.asarray(y_err,dtype=float)
        self.x_err=np.asarray(x_err,dtype=float)
        self.xy_err=np.asarray(xy_err,dtype=float)
        self.order=order
        self.ndim=order+2

    def ln_likelihood(self,pars):
        coeffs=pars[:-1]
        sigma_int=pars[-1]
        
        P=np.poly1d(coeffs)
        residual=self.y-P(self.x)
        variance=self.y_err**2 + sigma_int**2
        
        return -0.5*np.sum(np.log(2*np.pi*(variance)) + residual**2/variance)

    def ln_prior(self,pars):
        coeffs=pars[:-1]
        sigma_int=pars[-1]
        if np.any(np.abs(coeffs)>1e10):
            return -np.inf
        if sigma_int<0 or sigma_int>1e10:
            return -np.inf
        return 0.0

    def ln_posterior(self,pars):
        lp=self.ln_prior(pars)
        if not np.isfinite(lp):
            return -np.inf
        ll=self.ln_likelihood(pars)
        if not np.isfinite(ll):
            return -np.inf
        return lp+ll

    def __call__(self,pars):
        return self.ln_posterior(pars)


def hoggfit(x,y,xerr=None,yerr=None,xyerr=None,order=1,nwalkers=32,nsteps=10000):

    x = np.asarray(x,dtype=float)
    y = np.asarray(y,dtype=float)

    xerr = np.zeros_like(x) if xerr is None else np.asarray(xerr,dtype=float)
    yerr = np.zeros_like(y) if yerr is None else np.asarray(yerr,dtype=float)
    xyerr = np.zeros_like(x) if xyerr is None else np.asarray(xyerr,dtype=float)

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

    model = PolynomialModel(x, y, yerr, xerr, xyerr, order)

    ndim = model.ndim
    coeffs0 = np.polyfit(x, y, order)
    residual = y-np.polyval(coeffs0,x )
    sigma0 = np.std(residual)
    initial_params = np.zeros((nwalkers, ndim))

    for k in range(order+1):
        scale = 10*max(abs(coeffs0[k]), 1)
        initial_params[:, k] = coeffs0[k] + scale*np.random.randn(nwalkers)
    initial_params[:, -1] = sigma0*np.abs(1 + 0.1*np.random.randn(nwalkers))

    moves = [
            (emcee.moves.DESnookerMove(), 0.05), 
            (emcee.moves.DEMove(gamma0=1.2), 0.4), 
            (emcee.moves.StretchMove(a=5), 0.55)
        ]
    sampler=emcee.EnsembleSampler(nwalkers, ndim,model.ln_posterior, moves = moves)
    sampler.run_mcmc(initial_params, nsteps, progress=True)

    return sampler


def analyze_hoggfit(sampler,x,y,order=1,sigma_int_true=None,truths=None,burnin=250,thin=10,nsamp=200):

    chain=sampler.get_chain()
    samples=sampler.get_chain(discard=burnin,thin=thin,flat=True)
    samples_plot=samples.copy()

    labels=[rf"$a_{{{i}}}$" for i in range(order,-1,-1)]+[r"$\sigma_{\rm int}$"]

    fig,axes=plt.subplots(samples.shape[1],1,figsize=(10,2.2*samples.shape[1]),sharex=True)

    if samples.shape[1]==1:
        axes=[axes]

    for i,ax in enumerate(axes):
        ax.plot(chain[:,:,i],color="black",alpha=0.25,lw=0.5)
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()

    truths_plot=None
    if truths is not None:
        truths_plot=list(truths)
        if sigma_int_true is not None:
            truths_plot.append(sigma_int_true)
    fig=corner.corner(samples_plot,labels=labels,truths=truths_plot,show_titles=True,title_fmt=".3g",quantiles=[0.16,0.5,0.84],fill_contours=True,smooth=1.0,smooth1d=1.0,levels=(0.393,0.865,0.989),plot_datapoints=True)
    fig.set_size_inches(9,9)
    plt.show()

    inds=np.random.default_rng().choice(len(samples),size=min(nsamp,len(samples)),replace=False)
    x_dense=np.linspace(np.min(x),np.max(x),1000)
    curves=np.array([np.polyval(samples[i,:-1],x_dense) for i in inds])
    lower,upper=np.percentile(curves,[16,84],axis=0)

    fig,ax=plt.subplots(figsize=(8,6))
    ax.scatter(x,y,color="black",zorder=1)
    ax.fill_between(x_dense,lower,upper,color="orangered",alpha=0.3,label=r"1$\sigma$ CI")
    ax.plot(x_dense,np.median(curves,axis=0),color="orangered",lw=2,label="Median model")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid(alpha=0.15)
    ax.legend()

    plt.tight_layout()
    plt.show()

    q16,q50,q84=np.percentile(samples_plot,[16,50,84],axis=0)
    print("\nPosterior Results")
    print("-----------------")
    for i,label in enumerate(labels):
        print(f"{label:20s} = {q50[i]:.6e} (+{q84[i]-q50[i]:.6e}, -{q50[i]-q16[i]:.6e})")
        
    print("\nPosterior Covariance Matrix")
    print("--------------------------")
    covariance_matrix=np.cov(samples_plot,rowvar=False)
    print(covariance_matrix)


    return [q16,q50,q84],covariance_matrix
# ==================================================================== #




# LEGRENDRE POLYNOMIALS WITH TAYLOR APPROXIMATION
# ==================================================================== #
class PolynomialModel(object):

    def __init__(self,x,y,y_err,x_err,xy_err,order=1):
        self.x=np.asarray(x,dtype=float)
        self.y=np.asarray(y,dtype=float)
        self.y_err=np.asarray(y_err,dtype=float)
        self.x_err=np.asarray(x_err,dtype=float)
        self.xy_err=np.asarray(xy_err,dtype=float)
        self.order=order
        self.ndim=order+2
        self.xmin=np.min(self.x)
        self.xmax=np.max(self.x)


    def scale_x(self,x):
        return (2*x-(self.xmax+self.xmin))/(self.xmax-self.xmin)


    def legendre_model(self,x,coeffs):
        return np.polynomial.legendre.legval(self.scale_x(x),coeffs)


    def legendre_derivative(self,x,coeffs):
        z=self.scale_x(x)
        dcoeff=np.polynomial.legendre.legder(coeffs)
        return np.polynomial.legendre.legval(z,dcoeff)*2/(self.xmax-self.xmin)


    def ln_likelihood(self,pars):
        coeffs=pars[:-1]
        sigma_int=pars[-1]
        model_y=self.legendre_model(self.x,coeffs)
        slope=self.legendre_derivative(self.x,coeffs)
        residual=(self.y-model_y)/np.sqrt(1+slope**2)
        variance_perp=(self.y_err**2-2*slope*self.xy_err+slope**2*self.x_err**2)/(1+slope**2)+sigma_int**2
        return -0.5*np.sum(np.log(2*np.pi*variance_perp)+residual**2/variance_perp)


    def ln_prior(self,pars):
        coeffs=pars[:-1]
        sigma_int=pars[-1]
        if np.any(np.abs(coeffs)>1e10):
            return -np.inf
        if sigma_int<0 or sigma_int>1e10:
            return -np.inf
        return 0.0


    def ln_posterior(self,pars):
        lp=self.ln_prior(pars)
        if not np.isfinite(lp):
            return -np.inf
        ll=self.ln_likelihood(pars)
        if not np.isfinite(ll):
            return -np.inf
        return lp+ll


    def __call__(self,pars):
        return self.ln_posterior(pars)


def hoggfit(x,y,xerr=None,yerr=None,xyerr=None,order=1,nwalkers=32,nsteps=10000):

    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)

    xerr=np.zeros_like(x) if xerr is None else np.asarray(xerr,dtype=float)
    yerr=np.zeros_like(y) if yerr is None else np.asarray(yerr,dtype=float)
    xyerr=np.zeros_like(x) if xyerr is None else np.asarray(xyerr,dtype=float)

    mask=np.isfinite(x)&np.isfinite(y)&np.isfinite(xerr)&np.isfinite(yerr)&np.isfinite(xyerr)

    x=x[mask]
    y=y[mask]
    xerr=xerr[mask]
    yerr=yerr[mask]
    xyerr=xyerr[mask]

    idx=np.argsort(x)

    x=x[idx]
    y=y[idx]
    xerr=xerr[idx]
    yerr=yerr[idx]
    xyerr=xyerr[idx]

    model=PolynomialModel(x,y,yerr,xerr,xyerr,order)

    ndim=model.ndim
    z=model.scale_x(x)
    coeffs0=np.polynomial.legendre.legfit(z,y,order)
    residual=y-np.polynomial.legendre.legval(z,coeffs0)
    sigma0=np.std(residual)


    initial_params=np.zeros((nwalkers,ndim))
    for k in range(order+1):
        initial_params[:,k]=coeffs0[k]+0.05*np.random.randn(nwalkers)
    initial_params[:,-1]=sigma0*np.abs(1+0.05*np.random.randn(nwalkers))


    custom_move = emcee.moves.StretchMove(a=3.0)
    sampler=emcee.EnsembleSampler(nwalkers,ndim,model.ln_posterior, moves = custom_move)
    sampler.run_mcmc(initial_params,nsteps,progress=True)

    return sampler


def analyze_hoggfit(sampler,x,y,order=1,sigma_int_true=None,truths=None,burnin=250,thin=10,nsamp=200):

    chain=sampler.get_chain()
    samples=sampler.get_chain(discard=burnin,thin=thin,flat=True)

    xmin=np.min(x)
    xmax=np.max(x)

    def legendre_to_poly(coeffs):

        leg_poly=np.polynomial.legendre.Legendre(coeffs,domain=[xmin,xmax])
        poly=leg_poly.convert(kind=np.polynomial.Polynomial)

        return poly.coef[::-1]


    poly_samples=np.array([legendre_to_poly(s[:-1]) for s in samples])

    sigma_samples=samples[:,-1]

    full_samples=np.column_stack((poly_samples,sigma_samples))


    labels=[rf"$a_{{{i}}}$" for i in range(order,-1,-1)]
    labels.append(r"$\sigma_{\rm int}$")


    fig,axes=plt.subplots(full_samples.shape[1],1,figsize=(10,2.2*full_samples.shape[1]),sharex=True)

    if full_samples.shape[1]==1:
        axes=[axes]

    for i,ax in enumerate(axes):
        ax.plot(chain[:,:,i],color="black",alpha=0.25,lw=0.5)
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()



    truths_plot=None

    if truths is not None:

        truths_plot=list(truths)

        if sigma_int_true is not None:
            truths_plot.append(sigma_int_true)


    fig=corner.corner(full_samples,labels=labels,truths=truths_plot,show_titles=True,title_fmt=".3g",quantiles=[0.16,0.5,0.84],fill_contours=True,smooth=1.0,smooth1d=1.0,levels=(0.393,0.865,0.989),plot_datapoints=True)

    fig.set_size_inches(9,9)

    plt.show()



    inds=np.random.default_rng().choice(len(poly_samples),size=min(nsamp,len(poly_samples)),replace=False)


    x_dense=np.linspace(np.min(x),np.max(x),1000)


    curves=np.array([np.polyval(poly_samples[i],x_dense) for i in inds])


    lower,upper=np.percentile(curves,[16,84],axis=0)



    fig,ax=plt.subplots(figsize=(8,6))


    ax.scatter(x,y,color="black",zorder=1)

    ax.fill_between(x_dense,lower,upper,color="orangered",alpha=0.3,label=r"1$\sigma$ CI")

    ax.plot(x_dense,np.median(curves,axis=0),color="orangered",lw=2,label="Median model")


    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid(alpha=0.15)
    ax.legend()


    plt.tight_layout()

    plt.show()
    
    
    
    corr = np.corrcoef(samples, rowvar=False)

    corr_labels = [rf"$a_{{{i}}}$" for i in range(order,-1,-1)]
    corr_labels.append(r"$\sigma_{\rm int}$")

    fig,ax=plt.subplots(figsize=(8,7))
    im=ax.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_xticks(np.arange(len(corr_labels)))
    ax.set_yticks(np.arange(len(corr_labels)))

    ax.set_xticklabels(corr_labels,rotation=45,ha="right")
    ax.set_yticklabels(corr_labels)

    fig.colorbar(im,ax=ax,label="Correlation")

    ax.set_title("Posterior Correlation Matrix")

    plt.tight_layout()
    plt.show()



    q16,q50,q84=np.percentile(full_samples,[16,50,84],axis=0)


    print("\nPosterior Polynomial Coefficients")
    print("--------------------------------")

    for i,label in enumerate(labels):

        print(f"{label:20s} = {q50[i]:.6e} (+{q84[i]-q50[i]:.6e}, -{q50[i]-q16[i]:.6e})")


    print("\nPolynomial Equation")
    print("-------------------")

    equation=""

    for i,c in enumerate(q50[:-1]):

        power=order-i

        if power>1:
            equation+=f"{c:.6e}x^{power} + "

        elif power==1:
            equation+=f"{c:.6e}x + "

        else:
            equation+=f"{c:.6e}"


    print("y =",equation)


    print("\nIntrinsic Scatter")
    print("-----------------")
    print(f"sigma_int = {q50[-1]:.6e} (+{q84[-1]-q50[-1]:.6e}, -{q50[-1]-q16[-1]:.6e})")


    print("\nPosterior Covariance Matrix")
    print("---------------------------")

    covariance_matrix=np.cov(full_samples,rowvar=False)

    print(covariance_matrix)


    return [q16,q50,q84],covariance_matrix
# ==================================================================== #
