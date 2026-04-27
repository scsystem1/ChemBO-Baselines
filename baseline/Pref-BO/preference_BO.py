import sys,os,argparse
sys.path.append('src')
import numpy as np
import torch
from model import GP_Model,train_preference,compute_probability
from acquisition import optim, acquire
from sklearn.preprocessing import StandardScaler
from gpytorch.priors import GammaPrior

def load_data_npz(filepath):
    """
    Load data from a secure NumPy .npz file.
    
    Args:
        filepath: Path to the .npz file
        
    Returns:
        Tuple of loaded arrays
        
    Raises:
        ValueError: If file doesn't exist or validation fails
    """
    if not os.path.exists(filepath):
        raise ValueError(f"Data file not found: {filepath}")
    
    try:
        data = np.load(filepath, allow_pickle=False)  # Secure: no pickle allowed
        return data
    except Exception as e:
        raise ValueError(f"Failed to load data file {filepath}: {str(e)}")

def gen_preference_data(x,y,acc,m = 100,seed = 42):
    
    np.random.seed(seed)
    #first subsample the data
    all_idx = np.arange(x.shape[0])
    sub_idx = np.random.choice(all_idx,size = int(m/2),replace = False)
    x_train = x[sub_idx]
    y_train = y[sub_idx]
    
    #generate the labels based on this subsampled data
    train_comp = np.zeros((m,2),dtype = int)
    idxs = np.arange(x_train.shape[0])
    actions = [1 for i in range(int(np.round(acc*m)))] + [0 for i in range(int(np.round((1-acc)*m)))]
    actions = np.array(actions)
    np.random.shuffle(actions)

    for i in range(m):
        #choose 2 random points
        idx1,idx2 = np.random.choice(idxs,size = 2,replace = False)
        v1 = y_train[idx1]
        v2 = y_train[idx2]
        if actions[i] == 1:
            if v1 > v2:
                train_comp[i,0] = idx1
                train_comp[i,1] = idx2
            else:
                train_comp[i,0] = idx2
                train_comp[i,1] = idx1  
        else:
            if v1 < v2:
                train_comp[i,0] = idx1
                train_comp[i,1] = idx2
            else:
                train_comp[i,0] = idx2
                train_comp[i,1] = idx1

    return x_train, train_comp

def main(args):

    # Load data from secure NumPy format (no code execution risk)
    try:
        data = load_data_npz('data/dft.npz')
        all_x = data['x']
        all_y = data['y']
    except Exception as e:
        raise RuntimeError(
            f"Failed to load data/dft.npz: {str(e)}\n"
            f"Please convert your pickle files using: python convert_pickle_to_npz.py"
        )

    all_data = []
    for seed in range(int(args.n_trials)):
        np.random.seed(seed)

        all_idx = np.arange(all_x.shape[0])
        done_idx = np.random.choice(all_idx,size = int(args.n_start), replace = False) #idx of everything that is known
        exp_idx  = np.array([i for i in all_idx if i not in done_idx])                #idx of everything that is unknown

        #train the preference model
        # Load data from secure NumPy format (no code execution risk)
        try:
            data = load_data_npz('data/ohe.npz')
            px = data['x']
            py = data['y']
        except Exception as e:
            raise RuntimeError(
                f"Failed to load data/ohe.npz: {str(e)}\n"
                f"Please convert your pickle files using: python convert_pickle_to_npz.py"
            )

        x_pref,train_comp = gen_preference_data(px,py,acc = float(args.acc), m = int(args.m))  #this would be where we integrate the llm / survey

        prefm      = train_preference(x_train    = x_pref,
                                        train_comp = train_comp)

        all_pi     = compute_probability(model  = prefm,
                                            all_x  = px)

        #perform the BO for 50 rounds of acquisition
        n = 50
        args.fmax = np.max(all_y[done_idx])
        all_fmax  = np.zeros(n+1)
        all_fmax[0] = args.fmax

        for t in range(n):

            #scale the data as needed
            scaler = StandardScaler()
            y_train= all_y[done_idx].reshape(-1,1)
            if t != 0: y_train= scaler.fit_transform(y_train).flatten()
            args.fmax = np.max(y_train)

            #train the surrogate model
            x_train   = torch.tensor(all_x[done_idx],dtype = torch.float64)
            y_train   = torch.tensor(y_train,dtype = torch.float64)

            surrogate =        GP_Model(x_train, 
                                    y_train, 
                                    gpu=False,
                                    nu=2.5,
                                    noise_constraint=1e-5,
                                    lengthscale_prior=[GammaPrior(2.0, 0.2), 5.0],
                                    outputscale_prior=[GammaPrior(5.0, 0.5), 8.0],
                                    noise_prior=[GammaPrior(1.5, 0.5), 1.0],
                                    n_restarts=0,
                                    learning_rate=0.1,
                                    training_iters=100
                                    )

            surrogate.fit()

            #perform the acquisitio and adjust data structures as needed
            opt = optim(surrogate   = surrogate,
                        input_space = torch.tensor(all_x[exp_idx],dtype = torch.float64),
                        method      = args.method,
                        preference  = all_pi[exp_idx])

            tempid  = acquire(opt,args)
            new_idx = exp_idx[tempid]  #this is the id of the chosen experiment
            exp_idx  = np.delete(exp_idx,tempid)  
            done_idx = np.append(done_idx,new_idx)

            #adjust optimization things as needed
            args.iter += 1

            #record metrics as needed
            all_fmax[t+1] = np.max(all_y[done_idx])

        all_data.append(all_fmax)

    all_data = np.array(all_data)


    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate optimization of direct arylation parameters')
    parser.add_argument('method', help = 'acquisition method for BO')
    parser.add_argument('n_trials', help = 'how many repeated experiments')
    parser.add_argument('n_start', help = 'how many experiments to start BO with')
    parser.add_argument('output', help = 'name of the output file')

    #arguments for the plain BO acquisition strategies
    parser.add_argument('-fmax')

    #arguments for the human preference acquisition strategies
    parser.add_argument('-m',     help = 'Number of pairwise comparisons to do', default=650)
    parser.add_argument('-acc', help = 'A measure of the knowledge of the human', default = 1.0) 
    parser.add_argument('-beta',  help = 'Controlling how quickly the influence of the prior changes', default = 1)
    parser.add_argument('-iter',  help = 'Keeping track of the BO iteration', default = 1)

    



