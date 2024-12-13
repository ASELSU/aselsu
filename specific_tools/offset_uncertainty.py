import pandas as pd 
import numpy as np
import sys
from scipy.stats import t as t_coeff
from scipy import stats

def progress_bar(start, ii, end):
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" %('#'*round(100*(ii+1-start)/(end-start)),round(100*(ii+1-start)/(end-start))))
    sys.stdout.flush()

def compute_bias(gmsl_diff,ii,verbose=True):
    p = stats.shapiro(gmsl_diff)
        
    # calcul autocorrelation at lag1 
    values = pd.DataFrame(gmsl_diff)
    dataframe = pd.concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t-1', 't']
    result = dataframe.corr()

    # number of independant mesurment 
    rho1 = result['t-1']['t']
    n_sample = len(gmsl_diff[:ii])
    n = (1 - rho1)/(1 + rho1)*n_sample
    if n<=2: n=2 
    #if n>n_sample: n=n_sample
    # standard deviation 
    s = np.std(gmsl_diff[:ii])
    
    if p[1]>0.05: # non-Gaussian
        # Offset uncertainty 
        alpha = 0.32
        student_coeff = t_coeff.ppf(1-alpha/2,n-1)
        uncertainty = student_coeff * s/np.sqrt(n)
    
        if verbose:
            print("Shapiro-Wilk p value :", round(p[1],2))
            print("number of independant mesurment :",round(n))
            print("standard deviation :",round(s,2))
            print("Offset uncertainty (1σ) :",round(uncertainty,2))
    else: 
        uncertainty = s/np.sqrt(n) 
        
    return uncertainty,n
  