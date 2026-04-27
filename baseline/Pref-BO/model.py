import sys
import time
import torch
from copy import deepcopy
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from sklearn.linear_model import ARDRegression
from sklearn.model_selection import GridSearchCV
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from  gpytorch.distributions import MultivariateNormal
from gpytorch.priors import GammaPrior
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.fit import fit_gpytorch_mll

# Model from Shields 2021
####################

def to_torch(data, gpu=False):
    """
    Convert from pandas dataframe or numpy array to torch array.
    """
    
    if 'torch' in str(type(data)):
        torch_data = data
    
    else:
        try:
            torch_data = torch.from_numpy(np.array(data).astype('float')).float()
        except:
            torch_data = torch.tensor(data).float()

    if torch.cuda.is_available() and gpu == True:
        torch_data = torch_data.cuda()
    
    return torch_data

import numpy as np
class GP_Model:
    """Main gaussian process model used for Bayesian optimization.
    
    Provides a framework for specifiying exact GP models, hyperparameters, and 
    priors. This class also contains functions for training, sampling, forward 
    prediction, and variance estimation.
    """
    
    def __init__(self, X, y, training_iters=100, inference_type='MLE', 
                 learning_rate=0.1, noise_constraint=1e-5, gpu=False, nu=2.5,
                 lengthscale_prior=[GammaPrior(2.0, 0.2), 5.0], outputscale_prior=[GammaPrior(5.0, 0.5), 8.0],
                 noise_prior=[GammaPrior(1.5, 0.5), 1.0], n_restarts=0
                 ):
        """        
        Parameters
        ----------
        X : torch.tensor
            Training domain values.
        y : torch.tensor
            Training response values.
        training_iters : int
            Number of iterations to run ADAM optimizer durring training.
        inference_type : str
            Estimation procedue to be used. Currently only MLE is availible.
        learning_rate : float
            Learning rate for ADMA optimizer durring training.
        noise_constraint : float
            Noise is constrained to be positive. Set's the minimum noise level.
        gpu : bool 
            Use GPUs (if available) to run gaussian process computations. 
        nu : float 
            Matern kernel parameter. Options: 0.5, 1.5, 2.5.
        lengthscale_prior : [gpytorch.priors, init_value] 
            GPyTorch prior object and initial value. Sets a prior over length 
            scales.
        outputscale_prior : [gpytorch.priors, init_value] 
            GPyTorch prior object and initial value. Sets a prior over output
            scales.
        noise_prior : [gpytorch.priors, init_value]
            GPyTorch prior object and initial value. Sets a prior over output
            scales.
        n_restarts : int
            Number of random restarts for model training.
        
        Returns
        ----------
        None.
        """ 
        
        if inference_type == 'MCMC': print('Inference type not yet supported')
        
        # Initialization of main model components
        self.X = X
        self.y = y
        self.training_iters = training_iters
        self.inference_type = inference_type
        self.learning_rate = learning_rate
        self.noise_constraint = noise_constraint
        self.gpu = gpu  
        self.n_restarts = n_restarts
        self.lengthscale_prior = lengthscale_prior
        self.outputscale_prior = outputscale_prior
        self.noise_prior = noise_prior
        
        # Configure likelihood
        self.likelihood = GaussianLikelihood()
        if noise_prior != None:
            self.likelihood = GaussianLikelihood(noise_prior=noise_prior[0])
            self.likelihood.noise = torch.tensor([float(noise_prior[1])])
        
        # Set model
        self.model = gp_model(self.X, 
                              self.y, 
                              self.likelihood, 
                              gpu=gpu, 
                              nu=nu,
                              lengthscale_prior=lengthscale_prior,
                              outputscale_prior=outputscale_prior)
        
        # Set noise constraint
        self.model.likelihood.noise_covar.register_constraint(
		        "raw_noise", 
		        GreaterThan(noise_constraint)
		        )
        
        # GPU computation
        if torch.cuda.is_available() and gpu == True:
            self.model = self.model.cuda()
            
    # Maximum likelihood estimation
    def mle(self):
        """Uses maximum likelihood estimation to estimate model hyperparameters.
        
        """ 
        
        # Optimize MLL with user specified parameters
        loss = optimize_mll(self.model, self.likelihood, self.X, self.y, 
                     learning_rate=self.learning_rate, 
                     n_restarts=self.n_restarts,
                     training_iters=self.training_iters, 
                     noise_prior=self.noise_prior,
                     outputscale_prior=self.outputscale_prior, 
                     lengthscale_prior=self.lengthscale_prior)
        
        self.fit_restart_loss = loss
    
    # Fit model
    def fit(self):
        """Train the gaussian process model.""" 
        
        if self.inference_type == 'MLE':
            self.mle()
        else:
            print('Please specify valid inference type.')
            sys.exit(0)    
        
    # Mean of predictive posterior
    def predict(self, points):
        """Mean of gaussian process posterior predictive distribution.
        
        Parameters
        ----------
        points : torch.tensor
            Domain points to be evaluated.
        
        Returns
        ----------
        numpy.array
            Predicted response values for points.
        """ 
        
        # Get into evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Make predictions
        pred = self.model(points).mean.detach()
        
        return pred.numpy()
    
    # GP prediction variance
    def variance(self, points):
        """Variance of gaussian process posterior predictive distribution.
        
        Parameters
        ----------
        points : torch.tensor
            Domain points to be evaluated.
        
        Returns
        ----------
        numpy.array 
            Model variance a points.
        """
        
        # Get into evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Compuate variance
        var = self.model(points).variance.detach()
        
        if torch.cuda.is_available() and self.gpu == True:
            var = var.cpu()
        
        return var.numpy()
    
    # Sample posterior
    def sample_posterior(self, points, batch_size=1):
        """Sample functions from gaussian process posterior predictive distribution.
        
        Parameters
        ----------
        points : torch.tensor
            Domain points to be evaluated.
        batch_size : int
            Number of samples to draw.
        
        Returns
        ----------
        torch.tensor 
            Function values at points for samples.
        """
        
        points = to_torch(points, gpu=self.gpu)
        
        # Get into evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Sample the posterior
        posterior = self.model(points)
        samples = posterior.sample(torch.Size([batch_size]))
        
        return samples
    
class gp_model(ExactGP):
    """Base gaussian process model.
    
    GPyTorch's exact gaussian process regression with Matern kernel class.
    """
    
    def __init__(self, X, y, likelihood, gpu=False, nu=2.5,
                 lengthscale_prior=None, outputscale_prior=None
                 ):
        """        
        Parameters
        ----------
        X : torch.tensor
            Training domain values.
        y : torch.tensor 
            Training response values.
        likelihood : (gpytorch.likelihoods)
            Model likelihood.
        gpu : bool 
            Use GPUs (if available) to run gaussian process computations. 
        nu : float 
            Matern kernel parameter. Options: 0.5, 1.5, 2.5.
        lengthscale_prior : [gpytorch.priors, init_value] 
            GPyTorch prior object and initial value. Sets a prior over length 
            scales.
        outputscale_prior : [gpytorch.priors, init_value] 
            GPyTorch prior object and initial value. Sets a prior over output s
            cales.
        """
        
        super(gp_model, self).__init__(X, y, likelihood)
        
        # ARD
        num_dims = len(X) if len(X) == 0 else len(X[0])
        
        # Base kernel
        if lengthscale_prior == None:
            kernel = MaternKernel(nu=nu, 
                               ard_num_dims=num_dims)
        else:
            kernel = MaternKernel(nu=nu, 
                               ard_num_dims=num_dims,
                               lengthscale_prior=lengthscale_prior[0])
        
        # Mean
        self.mean_module = ConstantMean()
        
        # Output scale
        if outputscale_prior == None:
            self.covar_module = ScaleKernel(kernel)
        else:
            self.covar_module = ScaleKernel(
                                kernel,
                                outputscale_prior=outputscale_prior[0])
        
        # Set initial values
        if lengthscale_prior != None:
            try:
                ls_init = to_torch(lengthscale_prior[1], gpu=gpu)
                self.covar_module.base_kernel.lengthscale = ls_init
            except:
                uniform = to_torch(lengthscale_prior[1], gpu=gpu)
                ls_init = torch.ones(num_dims) * uniform
                self.covar_module.base_kernel.lengthscale = ls_init
            
        if outputscale_prior != None:
            os_init = to_torch(outputscale_prior[1], gpu=gpu)
            self.covar_module.outputscale = os_init
        
    # forward prediction
    def forward(self, x):
        """        
        Parameters
        ----------
        x : torch.tensor
            Domain points which define multivariate normal distribution.
        
        Returns
        ----------
        gpytorch.MultivariateNormal
            Multivariate normal distribution.
        """ 
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return MultivariateNormal(mean_x, covar_x) 
    

def build_dist_dict(noise_prior, outputscale_prior, lengthscale_prior):
    """
    Build a dictionary of distributions to sample for random restarts.
    """

    if noise_prior == None:
        noise_dist = GammaPrior(1.5,0.5)
    else:
        noise_dist = noise_prior[0]

    if outputscale_prior == None:
        output_dist = GammaPrior(3, 0.5)
    else:
        output_dist = outputscale_prior[0]
    
    if lengthscale_prior == None:
        lengthscale_dist = GammaPrior(3,0.5)
    else:
        lengthscale_dist = lengthscale_prior[0]

    distributions = {'likelihood.noise_covar.raw_noise': noise_dist,
                 'covar_module.raw_outputscale': output_dist,
                 'covar_module.base_kernel.raw_lengthscale': lengthscale_dist}
    
    return distributions

# Randomly set parmeters for model based on distributions

def set_init_params(dictionary, distributions, seed=0):
    """
    Generate a new random state dictionary with entries drawn from the list of 
    distributions.
    """
    
    dict_copy = deepcopy(dictionary)
    
    for key in distributions:
        
        # Get parameter values for dict entry
        params = dictionary[key]
        
        # Get distribution
        dist = distributions[key]
        
        # Generate inital points from distribution
        torch.manual_seed(seed)
        new_params = dist.expand(params.shape).sample().log()
        
        # Overwrite entry in copy
        dict_copy[key] = new_params
    
    return dict_copy

# Optimize a model via MLE

def optimize_mll(model, likelihood, X, y, learning_rate=0.1, 
                 n_restarts=0, training_iters=100, noise_prior=None,
                 outputscale_prior=None, lengthscale_prior=None):
    
    # Model and likelihood in training mode
    model.train()
    likelihood.train()
    
    # Use ADAM
    optimizer = torch.optim.Adam(
                [{'params': model.parameters()}, ],
                lr=learning_rate
                )
    # Marginal log likelihood loss                          
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Dictionary of distributions to draw random restarts from
    dist_dict = build_dist_dict(noise_prior, outputscale_prior, lengthscale_prior)
    
    # Restart optimizer with random inits drawn from priors
    states = []
    loss_list = []
    min_loss_list = []
    for restart in range(n_restarts + 1):
        
        step_losses = []
        # Optimization
        for i in range(training_iters):
            optimizer.zero_grad()
            output = model(X)
            loss = -mll(output, y)
            # Ensure loss is scalar for backward()
            if loss.numel() > 1:
                loss = loss.mean()
            step_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        states.append(deepcopy(mll.model.state_dict()))
        loss_list.append(step_losses)
        # Ensure loss is scalar for item()
        if loss.numel() > 1:
            loss = loss.mean()
        min_loss_list.append(loss.item())
    
        new_state = set_init_params(states[0], dist_dict, seed=restart)
        mll.model.load_state_dict(new_state)

    # Set to best state
    mll.model.load_state_dict(states[np.argmin(min_loss_list)])
    
    return loss_list

#######################

#Pairwise model implemented as Chu 2005

#preference model

def train_preference(x_train,train_comp, inp_transform = True):
    "x_train (N,d) and train_comp (M,2) are both numpy arrays"

    x_train = torch.as_tensor(x_train, dtype=torch.float64)
    train_comp = torch.as_tensor(train_comp).long()
    n_items, n_features = x_train.shape
    n_comparisons = train_comp.shape[0]
    dense_kernel_gib = (n_items * n_items * x_train.element_size()) / (1024 ** 3)

    print(
        f"[PrefBO] Fitting PairwiseGP with {n_items} items, "
        f"{n_features} features, {n_comparisons} comparisons",
        flush=True,
    )
    print(
        f"[PrefBO] PairwiseGP preflight: dtype={x_train.dtype}, "
        f"device={x_train.device}, torch_threads={torch.get_num_threads()}, "
        f"dense_kernel~{dense_kernel_gib:.2f} GiB per NxN matrix",
        flush=True,
    )
    if n_items >= 5000:
        print(
            "[PrefBO] PairwiseGP note: botorch initializes this model with "
            "SciPy fsolve on CPU, so large OCM runs can spend a long time in "
            "constructor setup before fit_gpytorch_mll starts.",
            flush=True,
        )

    if inp_transform:
        it = Normalize(d=x_train.shape[-1])
    else:
        it = None

    init_start = time.perf_counter()
    model = PairwiseGP(
        x_train,
        train_comp,
        input_transform=it,
    )
    init_seconds = time.perf_counter() - init_start
    print(f"[PrefBO] PairwiseGP constructor finished in {init_seconds:.1f}s", flush=True)
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_start = time.perf_counter()
    mll = fit_gpytorch_mll(mll)
    fit_seconds = time.perf_counter() - fit_start
    print(f"[PrefBO] PairwiseGP fit finished in {fit_seconds:.1f}s", flush=True)

    return model

def compute_probability(model,all_x, n_samples = 20000):
    "Computing the probability of the function maximum"

    #sample n_sample functions over the domain from the preference model posterior
    x_test = torch.tensor(all_x,dtype = torch.float64)
    all_samples = np.zeros((n_samples,x_test.shape[0]))
    mvn = model.posterior(x_test)
    for i in range(n_samples):
        all_samples[i,:] = mvn.sample().detach().numpy().flatten()

    #compute the probability of a point in the domain containing the function maximum
    p = np.zeros(x_test.shape[0])
    for curve in all_samples:
        idx = np.argmax(curve)
        p[idx]+=1
    p/= n_samples

    return p 

def compute_copeland(model,all_x, n_samples = 5000):
    
    x_test   = torch.tensor(all_x,dtype = torch.float64)
    copeland = np.zeros((n_samples,x_test.shape[0])) 
   
    mvn = model.posterior(x_test)
    for i in range(n_samples):
        sample = mvn.sample().detach().numpy().flatten()
        sorted_args = np.argsort(sample)
        for j,idx in enumerate(sorted_args[::-1]):
            copeland[i,idx] = j
            
    return copeland


