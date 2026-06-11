import GPy
import GPyOpt
import GPyOpt.core
import GPyOpt.models
import numpy as np
from GPyOpt.acquisitions import AcquisitionEI, AcquisitionMPI, AcquisitionLCB
from tqdm import tqdm

from BO import BO


def normalize(value):
    return (value - value.mean())/value.std()


def getProbs(scores, eta):
    dmax = scores.max()
    dmin = scores.min()
    if(dmax == dmin):
        aux = np.exp(np.zeros(scores.shape))
    else:
        aux = np.exp(eta * (scores - dmax)/(dmax-dmin))
    return  aux/aux.sum()

def chooseHedge(scores, eta):
    probs = getProbs(scores, eta)
    cumsum = probs.cumsum()
    aux = np.random.uniform(0, cumsum[-1])
    return np.argmax(cumsum > aux), probs

def build_acquisition(X_init, space, aquisition_function, model):
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space, eps=0)
    if(aquisition_function['type'] == 'ei'):
        aquisition_function = AcquisitionEI(model=model, space=space, optimizer=aquisition_optimizer,jitter=aquisition_function['epsilon'])
    elif(aquisition_function['type']== 'pi'):
        aquisition_function = AcquisitionMPI(model=model, space=space, optimizer=aquisition_optimizer,jitter=aquisition_function['epsilon'])
    elif(aquisition_function['type'] == 'lcb'):
        lcb_const = aquisition_function['upsilon']
        aquisition_function = AcquisitionLCB(model=model, space=space, optimizer=aquisition_optimizer,exploration_weight=lcb_const)
    elif(aquisition_function['type'] == 'mean'):
        aquisition_function = AcquisitionLCB(model=model, space=space, optimizer=aquisition_optimizer, exploration_weight=0.0)
    return aquisition_function


def build_bos(X_init, y_init, model, space, aquisition_functions):    
    bos = []
    for function in aquisition_functions:
        aquisition_function = build_acquisition(X_init, space, function, model)
        evaluator = GPyOpt.core.evaluators.Sequential(aquisition_function)
        bo = BO(model, space, None, aquisition_function, evaluator, X_init=X_init, Y_init=y_init)
        bos.append(bo)
    return bos

def get_best_evaluation(X_init, y_init, space, acquisitions, optimization_function, factor=1., iterations=10, eta=4, target=0):
    assert(X_init.shape[0] == y_init.shape[0])
    portfolio_size = len(acquisitions)
    scores_list=[]
    scores = np.zeros(portfolio_size)
    x = [None] * portfolio_size
    previous_x = [None] * portfolio_size

    X = X_init
    y = y_init

    acquisition_log = []

    pbar = tqdm(range(iterations), desc="BO Iterations")
    for i in pbar:
        kernel = GPy.kern.Matern52(input_dim=X.shape[1], ARD=False)
        model = GPyOpt.models.GPModel(kernel=kernel,optimize_restarts=5,verbose=False, ARD=False, exact_feval=True, max_iters=5000)
        normalized_y = normalize(y)
        model.updateModel(X,normalized_y,None,None)
        bos = build_bos(X, y, model, space, acquisitions)

        if(i!=0):
            for j in range(portfolio_size):
                previous_x[j] = x[j]

        for j in range(portfolio_size):
            x[j] = bos[j].suggest_next_locations(ignored_X=X)

        if(i!=0):
            for j in range(portfolio_size):
                scores[j] = factor*scores[j] - (model.predict(previous_x[j])[0]*y.std() + y.mean())
            scores_list.append(scores.copy())

        randomChoice, probs = chooseHedge(scores, eta)
        bestChoice = x[randomChoice]
        acquisition_log.append([i] + [acquisitions[randomChoice]['type']] + probs.tolist())

        x_next = bestChoice
        y_next = optimization_function(x_next)

        X = np.vstack((X, x_next))
        y = np.vstack((y, y_next))

        pbar.set_postfix(current_best=f"{y.min():.3f}")

        if y.min() <= target + 1e-10:
            print(f"\nTarget {target} reached at iteration {i+1}. Stopping optimization.")
            break
            

    execution = np.concatenate((X, y), axis=1)
    
    if previous_x[0] is not None:
        for j in range(portfolio_size):
            scores[j] = factor*scores[j] - (model.predict(previous_x[j])[0]*y.std() + y.mean())
        scores_list.append(scores.copy())

    return execution, scores_list, acquisition_log
