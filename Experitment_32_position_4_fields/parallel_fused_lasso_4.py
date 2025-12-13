from mpi4py import MPI
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import random
from numpy import linalg as LA
from scipy.linalg import svd
from scipy.sparse import kron, diags, vstack
from scipy.sparse.linalg import spsolve
from scipy import sparse
import numpy.linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Import your custom modules
from algorithm_solution_FL import solve_tv_problem, solve_cvxpy_problem_L1,solve_cvxpy_problem_L2,solve_combined_problem, DG_algorithm, bregman_algorithm_fused_lasso, regularization_for_l1, DG_algorithm_quadratic_solve_direct, fused_lasso_bregman_algorithm_con_solv
from functions_finite_diferences import create_gradient_operator
import pickle
import pyvista as pv

def parallel_hyperparameter_sweep():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Add debug output
    if rank == 0:
        print(f"Starting parallel execution with {size} processes")
    
    # Load data - all processes need this
    try:
        with open('new_try.pkl', 'rb') as f:      
            Data = pickle.load(f)
    except Exception as e:
        print(f"Rank {rank}: Error loading data: {e}")
        comm.Abort(1)
        return

    coordinates1 = Data['coordinates1']
    G_global_2_coils_2 = Data['G_gain']
    signals_exp_inside_3 = Data['signals_experimental']
    cj_3 = Data['concentration']

    num_v = 5

    # Parameter ranges
    '''Experiment 1 (3 faces 4 currents) of the big discretization'''

    ##fused lasso 
    alphas = np.logspace(-7,-1,10)
    betas =np.logspace(-3,5,10) 
    psis = np.logspace(8,15,10)

    # Create parameter combinations on root process and distribute
    if rank == 0:
        param_combinations = list(itertools.product(alphas, betas, psis))
        print(f"Total parameter combinations: {len(param_combinations)}")
        
        # Split work among processes
        chunks = [[] for _ in range(size)]
        for i, params in enumerate(param_combinations):
            chunks[i % size].append(params)
    else:
        chunks = None
        param_combinations = None

    # Scatter the work chunks to all processes
    local_combinations = comm.scatter(chunks, root=0)
    
    print(f"Rank {rank}: Processing {len(local_combinations)} parameter combinations")

    uinit = np.zeros(len(coordinates1))
    xMax = 0.4
    xMin = 0
    
    vector_d = np.sum(np.abs(G_global_2_coils_2), axis=0)
    Gamma = np.diag(vector_d)/np.linalg.norm(vector_d)


    mD = create_gradient_operator(num_v-1, num_v-1, num_v-1)
    
    

    # Process local combinations
    results = []
    
    for i, (alpha, beta, psi) in enumerate(local_combinations):
        try:
            # Add timeout or iteration limit if your solver supports it
            u_0 = solve_combined_problem(psi * G_global_2_coils_2, signals_exp_inside_3, mD, alpha, beta, xMax)[0]
            # u_0 = solve_cvxpy_problem_L1(psi * G_global_2_coils_2, signals_exp_inside_3, alpha, xMax)[0]
            # u_0 = solve_cvxpy_problem_L2(psi * G_global_2_coils_2, signals_exp_inside_3, alpha, xMax)[0]
            # u_0 = solve_tv_problem(psi * G_global_2_coils_2, signals_exp_inside_3, mD, alpha, xMax)[0]
            
            
            # Calculate metrics
            cc = np.corrcoef(u_0.flatten(), cj_3.flatten())[0, 1].item()
            RMSE_0 = LA.norm(cj_3 - u_0) / LA.norm(cj_3)
            
            results.append({
                'alpha': alpha,
                'beta': beta,
                'psi': psi,
                'cc': cc,
                'RMSE': RMSE_0,
                'u_0': u_0,
                'status': 'success'
            })
            
            if (i + 1) % 10 == 0:  # Print progress every 10 combinations
                print(f"Rank {rank}: Completed {i+1}/{len(local_combinations)} - "
                      f"α={alpha:.4e}, ψ={psi:.4e} → CC={cc:.4f}, RMSE={RMSE_0:.4f}")
        
        except Exception as e:
            results.append({
                'alpha': alpha,
                'beta': beta,
                'psi': psi,
                'cc': np.nan,
                'RMSE': np.nan,
                'u_0': None,
                'status': f'failed: {str(e)}'
            })
            
            print(f"Rank {rank}: Failed combination {i+1} - α={alpha:.4e}, ψ={psi:.4e}: {str(e)}")

    # Gather results with a timeout to prevent hanging
    print(f"Rank {rank}: Finished processing, gathering results...")
    
    try:
        all_results = comm.gather(results, root=0)
    except Exception as e:
        print(f"Rank {rank}: Error during gather: {e}")
        comm.Abort(1)
        return

    # Root process saves results
    if rank == 0:
        print("Root process: Gathering all results...")
        flat_results = [res for sublist in all_results for res in sublist]
        
        # Filter successful results
        successful_results = [res for res in flat_results if res['status'] == 'success']
       
        
        if successful_results:
            best_by_cc = max(successful_results, key=lambda x: x['cc'])
            best_by_rmse = min(successful_results, key=lambda x: x['RMSE'])
            
            # Save results
            output_data = {
                'best_by_cc': best_by_cc,
                'true_grouth': cj_3,
                'best_by_rmse': best_by_rmse,
                'total_combinations': len(param_combinations),
                'successful_combinations': len(successful_results)
            }
            
            with open('new_try_results_exp_1.pkl', 'wb') as f:
                pickle.dump(output_data, f)
            
            print("\n=== Best Successful Results ===")
            print(f"By CC (α={best_by_cc['alpha']:.4e}, β={best_by_cc['beta']:.4e}, "
                  f"ψ={best_by_cc['psi']:.4e}): CC={best_by_cc['cc']:.4f}, RMSE={best_by_cc['RMSE']:.4f}")
            
            print(f"By RMSE (α={best_by_rmse['alpha']:.4e}, β={best_by_rmse['beta']:.4e}, "
                  f"ψ={best_by_rmse['psi']:.4e}): CC={best_by_rmse['cc']:.4f}, RMSE={best_by_rmse['RMSE']:.4f}")
            
            print(f"Success rate: {len(successful_results)}/{len(param_combinations)} "
                  f"({len(successful_results)/len(param_combinations)*100:.1f}%)")
        else:
            print("\nWarning: No parameter combinations converged successfully!")
            with open('new_try_results_exp_1.pkl', 'wb') as f:
                pickle.dump({
                    'best_by_cc': None,
                    'true_grouth': cj_3,
                    'total_combinations': len(param_combinations),
                    'successful_combinations': 0
                }, f)
        
        print("Root process: Results saved successfully. Exiting.")

if __name__ == '__main__':
    # Add MPI finalize to ensure clean exit
    try:
        parallel_hyperparameter_sweep()
    except Exception as e:
        print(f"Main error: {e}")
        MPI.COMM_WORLD.Abort(1)
    finally:
        MPI.Finalize()
