from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
import warnings
from numpy import random
import pandas as pd
import scipy
from scipy import sparse
from scipy.sparse import kron, diags, vstack
from scipy.sparse.linalg import spsolve
warnings.filterwarnings('ignore')

import cvxpy as cp
import numpy as np




#algorithm and functions needed

def shrink(a, delta):
    return np.sign(a) * np.maximum(np.abs(a) - delta, 0)

# Functions for ADMM steps

def prox_l1(vX, lambdaFactor):
    # Soft Thresholding
    vX = np.maximum(vX - lambdaFactor, 0) + np.minimum(vX + lambdaFactor, 0)
    return vX


#######################################
######### Algorithms ##################
#######################################

def regularization_for_l1(G, uinit, V, alpha, xMax, xMin, parameters):
    residual = []
    relative_diff = []
    # beta = 0.9 * LA.norm(G.T.dot(G))
    # print(beta)
    beta=parameters['paramLambda']
    n_iteration = parameters['numIterations']
    
    # Compute the initial preproj_u value
    preproj_u = shrink(uinit - 2 * beta * G.T.dot(G.dot(uinit) - V), alpha * beta)
    u = np.maximum(np.minimum(preproj_u, xMax), xMin)
    
    for i in tqdm(range(n_iteration)):
        u_prev = u
        preproj_u = shrink(u - 2 * beta * G.T.dot(G.dot(u) - V), alpha * beta)
        u = np.maximum(np.minimum(preproj_u, xMax), xMin)
        residual.append(LA.norm(G.dot(u) - V))
        difference = LA.norm(u_prev - u)
        relative_diff.append(difference)
    
    return u, residual, relative_diff











def solve_combined_problem(G, V, D, lambda_1, lambda_2, C_max):
    """
    Solves:
        min_C (1/2)||G*C - V||_2^2 + lambda_1*||D*C||_1 + lambda_2*||C||_1
        subject to 0 <= C <= C_max

    Parameters:
        G: System matrix (m x n)
        V: Measurement vector (m x 1)
        lambda_1: TV regularization strength
        lambda_2: L1 regularization strength
        C_max: Upper bound on C
        n: Dimension of C

    Returns:
        C_opt: Optimal solution (n x 1)
        problem: CVXPY problem object
    """
    n = G.shape[1]
    C = cp.Variable(n)
    
    
    # Objective terms
    data_fidelity = 0.5 * cp.sum_squares(G @ C - V)
    tv_penalty = lambda_1 * cp.norm(D @ C, 1)  # TV term
    l1_penalty = lambda_2 * cp.norm(C, 1)      # L1 sparsity term
    objective = cp.Minimize(data_fidelity + tv_penalty + l1_penalty)
    
    # Constraints
    constraints = [C >= 0, C <= C_max]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, verbose=True)  # or solver=cp.SCS
    
    if problem.status != cp.OPTIMAL:
        raise Exception("Solver failed to converge.")
    
    return C.value, problem



#Total Variotion TV
def solve_tv_problem(G, V, D,lambda_val, C_max):
    """
    Solves the TV-regularized problem:
        min_C (1/2)||G*C - V||_2^2 + lambda*||D*C||_1s
        subject to 0 <= C <= C_max
    
    Parameters:
        G: System matrix (m x n)
        V: Measurement vector (m x 1)
        lambda_val: Regularization strength
        C_max: Upper bound on C
        n: Dimension of C (needed to construct D)
    
    Returns:
        C_opt: Optimal solution (n x 1)
        problem: CVXPY problem object
    """

    n = G.shape[1]
    # Variables
    C = cp.Variable(n)
    
    # Construct 1D finite difference matrix D (n-1 x n)
    
    # Objective
    data_fidelity = 0.5 * cp.sum_squares(G @ C - V)
    tv_penalty = lambda_val * cp.norm(D @ C, 1)  # TV = L1 norm of gradients
    # tv_penalty = lambda_val * cp.sum_squares(D @ C) #H1
    objective = cp.Minimize(data_fidelity + tv_penalty)
    
    # Constraints
    constraints = [C >= 0, C <= C_max]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)  # ECOS handles L1 terms well
    
    if problem.status != cp.OPTIMAL:
        raise Exception("Solver failed to converge.")
    
    return C.value, problem

#l1 norm
def solve_cvxpy_problem_L1(G, V, lambda_val, C_max):
    """
    Solves the optimization problem:
    min_C (1/2)||G*C - V||_2^2 + lambda*||C||_1
    subject to 0 <= C <= C_max
    
    Parameters:
        G: System matrix (m x n)
        V: Measurement vector (m x 1)
        lambda_val: Regularization parameter
        C_max: Upper bound for C
    
    Returns:
        C_opt: Optimal solution
        problem: The CVXPY problem object
    """
    n = G.shape[1]
    C = cp.Variable(n)
    
    # Define the objective function
    data_fidelity = 0.5 * cp.sum_squares(G @ C - V)
    l1_regularization = lambda_val * cp.norm(C, 1)  # L1 norm of V
    objective = cp.Minimize(data_fidelity + l1_regularization)
    
    # Define constraints
    constraints = [C >= 0, C <= C_max]
    
    # Form and solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)
    
    if problem.status != cp.OPTIMAL:
        raise Exception("Problem did not converge to optimal solution")
    
    return C.value, problem

#l2 norm
def solve_cvxpy_problem_L2(G, V, lambda_val, C_max, Matrix_sensitive = None):
    """
    Solves the optimization problem:
    min_C (1/2)||G*C - V||_2^2 + lambda*||C||_1
    subject to 0 <= C <= C_max
    
    Parameters:
        G: System matrix (m x n)
        V: Measurement vector (m x 1)
        lambda_val: Regularization parameter
        C_max: Upper bound for C
    
    Returns:
        C_opt: Optimal solution
        problem: The CVXPY problem object
    """
    n = G.shape[1]
    C = cp.Variable(n)

    if Matrix_sensitive is None:
        Matrix_sensitive = np.eye(n)
    
    
    # Define the objective function
    data_fidelity = 0.5 * cp.sum_squares(G @ C - V)
    l2_regularization = 0.5 *lambda_val* cp.sum_squares(Matrix_sensitive@C)  # L1 norm of V
    objective = cp.Minimize(data_fidelity + l2_regularization)
    
    # Define constraints
    constraints = [C >= 0, C <= C_max]
    
    # Form and solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)
    
    if problem.status != cp.OPTIMAL:
        raise Exception("Problem did not converge to optimal solution")
    
    return C.value, problem


    ####################################################33




def DG_algorithm_v2(G,C, V, D,alpha,xMax,xMin,sSolverParams,D2,vol_elements):
    '''algorithm of the paper bregman specific for finite elements of total variational norm of the paper'''
    
    print('inicio del proceso')
    vol_elements_3=np.concatenate([vol_elements, vol_elements,vol_elements], axis=0)
    diagonal_3=diags(vol_elements_3)
    diagonal_3_sqr=diags(vol_elements_3**0.5)
    

    lambda_p = sSolverParams['paramLambda']
    delta_p = sSolverParams['paramDelta']
    S=sSolverParams['S']
    beta_p = sSolverParams['paramBeta']
    numIterations = int(sSolverParams["numIterations"])
    
    # Define the size of vX
    # num_rows = vY.shape[0]
    num_col=G.shape[1]

    # Array for residual evolution
    residual = []
    error_values=[]
    tv_norms=[]
    costs=[]
    d_terms=[]
    box_terms=[]

    # Initialize mX with zeros
    # mX = np.zeros((num_rows, numIterations))
    Id_ec= sparse.eye(num_col)

    # Calculate mC using Cholesky decomposition
    if sSolverParams['lasso']==True:
        mC = G.T.dot(G)+lambda_p *S* (D2.T.dot(D))+(delta_p+beta_p)*Id_ec
    else:
        print('TV norm')
        mC = G.T.dot(G)+lambda_p *S* (D2.T.dot(D))+delta_p*Id_ec 
  
    d = shrink(D @ C, alpha / (lambda_p*S))
    #vZ = prox_l1(mD @ vX, paramLambda / lambda_p)
    
    # print('inicio del proceso')
    b = D.dot(C) - d
    
    y=np.maximum(np.minimum(C,xMax),xMin)
    w=np.zeros(num_col)

    # Set the first column of mX to vX
    # mX[:, 0] = vX

    print('inicio del for')
   
    for ii in tqdm(range(1, numIterations)):
        b_rs = G.T.dot(V) + lambda_p *S* D2.T.dot(d - b)+delta_p*(y+w)
        C = spsolve(mC, b_rs)
        #vXapr=solver_regu_problem(A,vXapr, vY, paramRhoval,vZ,vU,mD,500100)
        d = shrink(D.dot(C) + b, alpha/ (lambda_p*S))

        #vZ = prox_l1(mD.dot(vX) + vU, paramLambda / paramRhoval)
        res = D.dot(C) - d #difference between gradient
        b = b + res
        residual.append(LA.norm(res))
        
        w=w+y-C
        y=np.maximum(np.minimum(C,xMax),xMin)

        f=LA.norm(G.dot(C)-V)**2
        error_values.append(f)
        tvnorm=LA.norm(diagonal_3.dot(d),1)
        tv_norms.append(tvnorm)

        d_term=LA.norm(diagonal_3_sqr.dot(b+res))**2
        d_terms.append(d_term)

        box_term=LA.norm(C-y-w)**2
        box_terms.append(box_term)
        cost = 0.5*f+ alpha*tvnorm+0.5*lambda_p*S*d_term+0.5*delta_p*box_term
        costs.append(cost) 

    return C, residual,d,b,error_values,d_terms,tv_norms, costs







        


def DG_algorithm_conjugated_gradient_method(G,C, V, D,alpha,xMax,xMin,sSolverParams,D2,max_its=50):
    '''
    Using split bregman algorithm and the gradient conjugated method for the quadratic constraint  optimization problem
    it does not solve the system directly then avoid the use of the parameter delta
    '''
    print('inicio del proceso')

    lambda_p = sSolverParams['paramLambda']
    numIterations = int(sSolverParams["numIterations"])
    
    # Define the size of vX
    # num_rows = vY.shape[0]
    num_col=G.shape[1]

    # Array for residual evolution
    residual = []
    error_values=[]

    # Initialize mX with zeros
    # mX = np.zeros((num_rows, numIterations))
    
    mC = G.T.dot(G)+lambda_p* (D2.T.dot(D))


    # initial conditions
    # Calculate vZ and vU
    d = shrink(D @ C, alpha / (lambda_p))
    #vZ = prox_l1(mD @ vX, paramLambda / lambda_p)
    
    # print('inicio del proceso')
    b = D.dot(C) - d

    # Set the first column of mX to vX
    # mX[:, 0] = vX

    print('inicio del for')
   
    for _ in tqdm(range(1, numIterations)):
        b_rs = G.T.dot(V) + lambda_p * D2.T.dot(d - b)

        solver = GPCGSolver(mC,b_rs,xMin*np.ones(num_col),xMax*np.ones(num_col))
        C = solver.solve(maxits=max_its)['x']

        d = shrink(D.dot(C) + b, alpha/ (lambda_p))

        #vZ = prox_l1(mD.dot(vX) + vU, paramLambda / paramRhoval)
        res = D.dot(C) - d #difference between gradient
        b = b + res
        residual.append(LA.norm(res))
        
        
        f=LA.norm(G.dot(C)-V)**2
        error_values.append(f)
    return C, residual,d,b,error_values












def DG_algorithm(G,C, V, D,alpha,xMax,xMin,sSolverParams):
    
    print('Ususally SB algorithm for any kinfd of structure in norm 1')

    lambda_p = sSolverParams['paramLambda']*sSolverParams['S']
    delta_p = sSolverParams['paramDelta']
    beta_p = sSolverParams['paramBeta']
    numIterations = int(sSolverParams["numIterations"])
    
    # Define the size of vX
    # num_rows = vY.shape[0]
    num_col=G.shape[1]

    # Array for residual evolution
    residual = []
    error_values=[]
    tv_norms=[]
    costs=[]
    d_terms=[]
    box_terms=[]

    # Initialize mX with zeros
    # mX = np.zeros((num_rows, numIterations))
    Id_ec= sparse.eye(num_col)

    # Calculate mC using Cholesky decomposition
    if sSolverParams['lasso']==True:
        mC = G.T.dot(G)+lambda_p * (D.T.dot(D))+(delta_p+beta_p)*Id_ec
    else:
        print('l1 norm')
        mC = G.T.dot(G)+lambda_p * (D.T.dot(D))+delta_p*Id_ec 
    # mC = np.linalg.cholesky(prod)
    # mC = cholesky(mI + paramRho * (mD.T @ mD), lower=True)


    # initial conditions
    # Calculate vZ and vU
    d = shrink(D @ C, alpha / lambda_p)
    #vZ = prox_l1(mD @ vX, paramLambda / lambda_p)
    
    # print('inicio del proceso')
    b = D.dot(C) - d
    
    y=np.maximum(np.minimum(C,xMax),xMin)
    w=np.zeros(num_col)

    # Set the first column of mX to vX
    # mX[:, 0] = vX

    print('inicio del for')
   
    for ii in tqdm(range(1, numIterations)):
        b_rs = G.T.dot(V) + lambda_p * D.T.dot(d - b)+delta_p*(y+w)
        C = spsolve(mC, b_rs)
        #vXapr=solver_regu_problem(A,vXapr, vY, paramRhoval,vZ,vU,mD,500100)
        d = shrink(D.dot(C) + b, alpha/ lambda_p)

        #vZ = prox_l1(mD.dot(vX) + vU, paramLambda / paramRhoval)
        res = D.dot(C) - d #difference between gradient
        b = b + res
        residual.append(LA.norm(res))
        
        w=w+y-C
        y=np.maximum(np.minimum(C,xMax),xMin)

        f=LA.norm(G.dot(C)-V)**2
        error_values.append(f)


        tvnorm=LA.norm(d,1)
        tv_norms.append(tvnorm)

        d_term=LA.norm(b+res)**2
        d_terms.append(d_term)

        box_term=LA.norm(C-y-w)**2
        box_terms.append(box_term)



        cost = 0.5*f+ alpha*tvnorm+0.5*lambda_p*d_term+0.5*delta_p*box_term
        costs.append(cost) 
    return C, residual,d,b,error_values,d_terms,box_terms, costs




def bregman_algorithm_fused_lasso(A,vX, vY, mD,alpha,beta,xMax,xMin,sSolverParams):
    
    print('inicio del proceso')
    lambda_p_1 = sSolverParams['paramLambda1']
    lambda_p_2 = sSolverParams['paramLambda2']
    delta_p = sSolverParams['paramDelta'] #parameters of constrain box
    numIterations = int(sSolverParams["numIterations"])

    num_col=A.shape[1]

    # Array for residual evolution
    residual1 = []
    residual2 = []
    error_values=[]

    # Create an identity matrix mI
    mIc= sparse.eye(num_col)

    # Calculate mC
    mC = A.T.dot(A)+lambda_p_1 * (mD.T.dot(mD))+(lambda_p_2+delta_p)*mIc 

    # initial conditions
    # Calculate vZ and vU
    vZ = shrink(mD @ vX, alpha / lambda_p_1)
    #vZ = prox_l1(mD @ vX, paramLambda / lambda_p_1)
    

    vW_ini= shrink(mIc @ vX, beta / lambda_p_2)
    
    vU = mD @ vX - vZ

    vW=  vX-vW_ini

    
    

    y=np.maximum(np.minimum(vX,xMax),xMin)
    w=np.zeros(num_col)


    print('inicio del for')
   
    for ii in tqdm(range(1, numIterations)):
        b = A.T.dot(vY) + lambda_p_1 * mD.T.dot(vZ - vU)+ lambda_p_2 * (vW_ini - vW)+delta_p*(y+w) 
        vX = spsolve(mC, b)
        #vXapr=solver_regu_problem(A,vXapr, vY, paramRhoval,vZ,vU,mD,500100)
        
        # vZ = shrink(mD.dot(vX) + vU/ lambda_p_1, alpha/ lambda_p_1)
        vZ = shrink(mD.dot(vX) + vU, alpha/ lambda_p_1)
        res1 = mD.dot(vX) - vZ
        # vU = vU + lambda_p_1*res1
        vU = vU + res1
        residual1.append(LA.norm(res1))

        # vW_ini = shrink(vX + vW/lambda_p_2, beta/lambda_p_2)
        vW_ini = shrink(vX + vW, beta/lambda_p_2)
        res2 = vX - vW_ini
        # vW = vW + lambda_p_2*res2
        vW = vW + res2

        residual2.append(LA.norm(res2))
        error_values.append(LA.norm(A.dot(vX)-vY))
        
        w=w+y-vX
        y=np.maximum(np.minimum(vX,xMax),xMin)

        # mX[:, ii] = vX

    # return vX, mX, residual
    return vX, residual1,residual2,vZ,vU,error_values


#################################################################
###############algoritms directs  with library cvxp
#######################################################


def solve_cvxpy_problem(G, V, A, d_k, b_k, lambda_val, C_max):
    """
    Solves the optimization problem:
    min_C (1/2)||G*C - V||_2^2 + (lambda/2)||d^k - A*C - b^k||_2^2
    subject to 0 <= C <= C_max
    
    Parameters:
        G: System matrix (m x n)
        V: Measurement vector (m x 1)
        A: Regularization operator matrix (p x n)
        d_k: Current auxiliary variable (p x 1)
        b_k: Current Bregman parameter (p x 1)
        lambda_val: Regularization parameter
        C_max: Upper bound for C
    
    Returns:
        C_opt: Optimal solution
        problem: The CVXPY problem object
    """
    n = G.shape[1]
    C = cp.Variable(n)
    
    # Define the objective function
    data_fidelity = 0.5 * cp.sum_squares(G @ C - V)
    regularization = 0.5 * lambda_val * cp.sum_squares(A @ C + b_k - d_k )
    objective = cp.Minimize(data_fidelity + regularization)
    
    # Define constraints
    constraints = [C >= 0, C <= C_max]
    
    # Form and solve problem
    problem = cp.Problem(objective, constraints)
    # problem.solve(verbose=True)  # Remove verbose for production
    problem.solve() 
    if problem.status != cp.OPTIMAL:
        raise Exception("Problem did not converge to optimal solution")
    
    return C.value, problem








def DG_algorithm_quadratic_solve_direct(G,C, V, D,alpha,xMax,xMin,sSolverParams):
    '''
    Using split bregman algorithm and the gradient conjugated method for the quadratic constraint  optimization problem
    it does not solve the system directly then avoid the use of the parameter delta
    '''
    print('inicio del proceso')

    lambda_p = sSolverParams['paramLambda']*sSolverParams['S']
    numIterations = int(sSolverParams["numIterations"])
    
    # Define the size of vX
    # num_rows = vY.shape[0]
    # num_col=G.shape[1]

    # Array for residual evolution
    residual = []
    error_values=[]

    # Initialize mX with zeros
    # mX = np.zeros((num_rows, numIterations))
    
    # mC = G.T.dot(G)+lambda_p* (D.T.dot(D))


    # initial conditions
    # Calculate vZ and vU
    d = shrink(D @ C, alpha / (lambda_p))
    #vZ = prox_l1(mD @ vX, paramLambda / lambda_p)
    
    # print('inicio del proceso')
    b = D.dot(C) - d

    # Set the first column of mX to vX
    # mX[:, 0] = vX

    print('inicio del for')
   
    for _ in tqdm(range(1, numIterations)):

        C_opt, problem = solve_cvxpy_problem(G, V, D, d, b, lambda_p, xMax)

        C = C_opt



        d = shrink(D.dot(C) + b, alpha/ (lambda_p))

        #vZ = prox_l1(mD.dot(vX) + vU, paramLambda / paramRhoval)
        res = D.dot(C) - d #difference between gradient
        b = b + res
        residual.append(LA.norm(res))
        
        
        f=LA.norm(G.dot(C)-V)**2
        error_values.append(f)
    return C, residual,d,b,error_values


#####  split bregman method for FEM

def solve_cvxpy_problem_v2(G, V, A, d_k, b_k, T,lambda_val, C_max):
    """
    Solves the optimization problem:
    min_C (1/2)||G*C - V||_2^2 + (lambda/2)||T^0.5(d^k - A*C - b^k)||_2^2
    subject to 0 <= C <= C_max
    
    Parameters:
        G: System matrix (m x n)
        V: Measurement vector (m x 1)
        A: Regularization operator matrix (p x n)
        d_k: Current auxiliary variable (p x 1)
        b_k: Current Bregman parameter (p x 1)
        lambda_val: Regularization parameter
        C_max: Upper bound for C
    
    Returns:
        C_opt: Optimal solution
        problem: The CVXPY problem object
    """
    n = G.shape[1]
    C = cp.Variable(n)
    
    # Define the objective function
    data_fidelity = 0.5 * cp.sum_squares(G @ C - V)
    regularization = 0.5 * lambda_val * cp.sum_squares(T@(A @ C + b_k - d_k) )
    objective = cp.Minimize(data_fidelity + regularization)
    
    # Define constraints
    constraints = [C >= 0, C <= C_max]
    
    # Form and solve problem
    problem = cp.Problem(objective, constraints)
    # problem.solve(verbose=True)  # Remove verbose for production
    problem.solve() 
    if problem.status != cp.OPTIMAL:
        raise Exception("Problem did not converge to optimal solution")
    
    return C.value, problem



def DG_algorithm_v2_direct(G,C, V, D,alpha,xMax,xMin,sSolverParams,vol_elements):
    '''algorithm of the paper bregman specific for finite elements of total variational norm of the paper'''
    
    print('inicio del proceso')
    vol_elements_3=np.concatenate([vol_elements, vol_elements,vol_elements], axis=0)
    diagonal_3=diags(vol_elements_3)
    diagonal_3_sqr=diags(vol_elements_3**0.5)
    

    lambda_p = sSolverParams['paramLambda']
    numIterations = int(sSolverParams["numIterations"])
    
    # Define the size of vX
    # num_rows = vY.shape[0]
    num_col=G.shape[1]

    # Array for residual evolution
    residual = []
    error_values=[]

    # Initialize mX with zeros
    # mX = np.zeros((num_rows, numIterations))
    Id_ec= sparse.eye(num_col)

   
  
    d = shrink(D @ C, alpha / (lambda_p))
    #vZ = prox_l1(mD @ vX, paramLambda / lambda_p)
    
    # print('inicio del proceso')
    b = D.dot(C) - d
    
    

    print('inicio del for')
   
    for ii in tqdm(range(1, numIterations)):


        C_opt, problem = solve_cvxpy_problem_v2(G, V, D, d, b, diagonal_3_sqr,lambda_p, xMax)

        C = C_opt

        #vXapr=solver_regu_problem(A,vXapr, vY, paramRhoval,vZ,vU,mD,500100)
        d = shrink(D.dot(C) + b, alpha/ (lambda_p))

        #vZ = prox_l1(mD.dot(vX) + vU, paramLambda / paramRhoval)
        res = D.dot(C) - d #difference between gradient
        b = b + res
        residual.append(LA.norm(res))
        
    

        f=LA.norm(G.dot(C)-V)**2
        error_values.append(f)
       
    return C, residual,d,b,error_values



###fused lasso


def solve_cvxpy_problem_fused_lasso(G, V, A, I, d1_k, b1_k, d2_k, b2_k, lambda_val1,lambda_val2, C_max):
    """
    Solves the optimization problem:
    min_C (1/2)||G*C - V||_2^2 + (lambda/2)||d1^k - A*C - b1^k||_2^2 + (lambda/2)||d2^k - I*C - b2^k||_2^2
    subject to 0 <= C <= C_max
    
    Parameters:
        G: System matrix (m x n)
        V: Measurement vector (m x 1)
        A: Regularization operator matrix (p x n)
        d_k: Current auxiliary variable (p x 1)
        b_k: Current Bregman parameter (p x 1)
        lambda_val1: Regularization parameter
        lambda_val2: Regularization parameter
        C_max: Upper bound for C
    
    Returns:
        C_opt: Optimal solution
        problem: The CVXPY problem object
    """
    n = G.shape[1]
    C = cp.Variable(n)
    
    # Define the objective function
    data_fidelity = 0.5 * cp.sum_squares(G @ C - V)
    regularization1 = 0.5 * lambda_val1 * cp.sum_squares(A @ C + b1_k - d1_k )
    regularization2 = 0.5 * lambda_val2 * cp.sum_squares(I @ C + b2_k - d2_k )

    objective = cp.Minimize(data_fidelity + regularization1 + regularization2)
    
    # Define constraints
    constraints = [C >= 0, C <= C_max]
    
    # Form and solve problem
    problem = cp.Problem(objective, constraints)
    # problem.solve(verbose=True)  # Remove verbose for production
    problem.solve() 
    if problem.status != cp.OPTIMAL:
        raise Exception("Problem did not converge to optimal solution")
    
    return C.value, problem



def fused_lasso_bregman_algorithm_con_solv(A,vX, vY, mD,alpha,beta,xMax,xMin,sSolverParams):
    
    print('inicio del proceso')
    lambda_p_1 = sSolverParams['paramLambda1']
    lambda_p_2 = sSolverParams['paramLambda2']
    numIterations = int(sSolverParams["numIterations"])

    num_col=A.shape[1]

    # Array for residual evolution
    residual1 = []
    residual2 = []
    error_values=[]

    # Create an identity matrix mI
    mIc= sparse.eye(num_col)

    # # Calculate mC
    # mC = A.T.dot(A)+lambda_p_1 * (mD.T.dot(mD))+(lambda_p_2+delta_p)*mIc 

    # initial conditions
    # Calculate vZ and vU
    vZ = shrink(mD @ vX, alpha / lambda_p_1)
    #vZ = prox_l1(mD @ vX, paramLambda / lambda_p_1)
    

    vW_ini= shrink(mIc @ vX, beta / lambda_p_2)
    
    vU = mD @ vX - vZ

    vW=  vX-vW_ini

    
    

    # y=np.maximum(np.minimum(vX,xMax),xMin)
    # w=np.zeros(num_col)


    print('inicio del for')
   
    for ii in tqdm(range(1, numIterations)):

        vX, problem = solve_cvxpy_problem_fused_lasso(A, vY, mD, mIc, vZ, vU, vW_ini, vW, lambda_p_1,lambda_p_2, xMax)

        #vXapr=solver_regu_problem(A,vXapr, vY, paramRhoval,vZ,vU,mD,500100)
        
        # vZ = shrink(mD.dot(vX) + vU/ lambda_p_1, alpha/ lambda_p_1)
        vZ = shrink(mD.dot(vX) + vU, alpha/ lambda_p_1)
        res1 = mD.dot(vX) - vZ
        # vU = vU + lambda_p_1*res1
        vU = vU + res1
        residual1.append(LA.norm(res1))

        # vW_ini = shrink(vX + vW/lambda_p_2, beta/lambda_p_2)
        vW_ini = shrink(vX + vW, beta/lambda_p_2)
        res2 = vX - vW_ini
        # vW = vW + lambda_p_2*res2
        vW = vW + res2

        residual2.append(LA.norm(res2))
        error_values.append(LA.norm(A.dot(vX)-vY))


        # mX[:, ii] = vX

    # return vX, mX, residual
    return vX, residual1,residual2,vZ,vU,error_values




















#######################################################
####################### extra algorithms ######################
#######################################################

def bregman_algorithm_fused_lasso_2(A,vX, vY, mD,alpha,beta,xMax,xMin,sSolverParams):
    
    print('inicio del proceso')
    lambda_p_1 = sSolverParams['paramLambda1']
    lambda_p_2 = sSolverParams['paramLambda2']
    delta_p = sSolverParams['paramDelta'] #parameters of constrain box
    numIterations = int(sSolverParams["numIterations"])

    num_col=A.shape[1]

    # Array for residual evolution
    residual1 = []
    residual2 = []
    error_values=[]

    tv_norm=[]
    costs=[]


    # Create an identity matrix mI
    mIc= sparse.eye(num_col)

    # Calculate mC using Cholesky decomposition
    mC = A.T.dot(A)+lambda_p_1 * (mD.T.dot(mD))+(lambda_p_2+delta_p)*mIc 

    # initial conditions
    # Calculate vZ and vU
    vZ = shrink(mD @ vX, alpha / lambda_p_1)
    

    vW_ini= shrink(vX, beta / lambda_p_2)
    
    
    vW=vX-vW_ini

    vU = mD @ vX - vZ
    

    w=np.zeros(num_col)


    print('inicio del for')
   
    for _ in tqdm(range(1, numIterations)):

        d=np.maximum(np.minimum(vX-w,xMax),xMin)


        b = A.T.dot(vY) + lambda_p_1 * mD.T.dot(vZ - vU)+ lambda_p_2 * (vW_ini - vW)+delta_p*(np.maximum(np.minimum(d-w,xMax),xMin)+w) 
        vX = spsolve(mC, b)
        #vXapr=solver_regu_problem(A,vXapr, vY, paramRhoval,vZ,vU,mD,500100)
        
        # vZ = shrink(mD.dot(vX) + vU/ lambda_p_1, alpha/ lambda_p_1)
        vZ = shrink(mD.dot(vX) + vU, alpha/ lambda_p_1)
        res1 = mD.dot(vX) - vZ
        # vU = vU + lambda_p_1*res1
        vU = vU + res1
        residual1.append(LA.norm(res1))

        # vW_ini = shrink(vX + vW/lambda_p_2, beta/lambda_p_2)
        vW_ini = shrink(vX + vW, beta/lambda_p_2)
        res2 = vX - vW_ini
        # vW = vW + lambda_p_2*res2
        vW = vW + res2
        residual2.append(LA.norm(res2))
        
        w=w+np.maximum(np.minimum(d,xMax),xMin)-vX

        error_values.append(LA.norm(A.dot(vX)-vY))


        f=LA.norm(A.dot(vX)-vY)**2
        tvnorm=LA.norm(mD.dot(vX))


        error_values.append(f)
        tv_norm.append(tvnorm)




        cost = 0.5*f+ alpha*tvnorm
        costs.append(cost) 


        # mX[:, ii] = vX

    # return vX, mX, residual
    return vX, residual1,residual2,vZ,vU, error_values,costs



def bregman_algorithm_fused_lasso_3(A,vX, vY, mD,alpha,beta,xMax,xMin,sSolverParams):
    
    print('inicio del proceso')
    lambda_p_1 = sSolverParams['paramLambda1']
    lambda_p_2 = sSolverParams['paramLambda2']

    delta_p_1=sSolverParams['paramdelta1']
    delta_p_2=sSolverParams['paramdelta2']


    delta_p = sSolverParams['paramDelta'] #parameters of constrain box
    numIterations = int(sSolverParams["numIterations"])

    num_col=A.shape[1]

    # Array for residual evolution
    residual1 = []
    residual2 = []
    error_values=[]
    tv_norm=[]
    costs=[]

    # Create an identity matrix mI
    mIc= sparse.eye(num_col)

    # Calculate mC using Cholesky decomposition
    mC = A.T.dot(A)+lambda_p_1 * (mD.T.dot(mD))+(lambda_p_2+delta_p)*mIc 

    # initial conditions
    # Calculate vZ and vU
    vZ = shrink(mD @ vX, alpha / lambda_p_1)
    

    vW_ini= shrink(vX, beta / lambda_p_2)
    
    
    vW=vX-vW_ini

    vU = mD @ vX - vZ
    

    y=np.maximum(np.minimum(vX,xMax),xMin)
    w=np.zeros(num_col)


    print('inicio del for')
   
    for _ in tqdm(range(1, numIterations)):

        d=np.maximum(np.minimum(vX-w,xMax),xMin)


        b = A.T.dot(vY) + lambda_p_1 * mD.T.dot(vZ - vU/lambda_p_1)+ lambda_p_2 * (vW_ini - vW/lambda_p_2)+delta_p*(np.maximum(np.minimum(d-w,xMax),xMin)+w) 
        vX = spsolve(mC, b)
        #vXapr=solver_regu_problem(A,vXapr, vY, paramRhoval,vZ,vU,mD,500100)
        
        vZ = shrink(mD.dot(vX) + vU/ lambda_p_1, alpha/ lambda_p_1)
        
        vW_ini = shrink(vX + vW/lambda_p_2, beta/lambda_p_2)

        res1 = mD.dot(vX) - vZ
        vU = vU + delta_p_1*res1
    
        residual1.append(LA.norm(res1))

   
        res2 = vX - vW_ini
        vW = vW + delta_p_2*res2
        
        residual2.append(LA.norm(res2))
        
        w=w+np.maximum(np.minimum(d,xMax),xMin)-vX
        f=LA.norm(A.dot(vX)-vY)**2
        tvnorm=LA.norm(mD.dot(vX))


        error_values.append(f)
        tv_norm.append(tvnorm)
        cost = 0.5*f+ alpha*tvnorm
        costs.append(cost) 

        # mX[:, ii] = vX

    # return vX, mX, residual

    print(LA)

    return vX, residual1,residual2,vZ,vU, error_values,costs




def bregman_method_sensitive_1(A,vX, vY, mD,sensetive_diag,alpha,beta,xMax,xMin,sSolverParams):
    
    print('inicio del proceso')
    lambda_p_1 = sSolverParams['paramLambda1']
    lambda_p_2 = sSolverParams['paramLambda2']
    delta_p = sSolverParams['paramDelta'] #parameters of constrain box
    numIterations = int(sSolverParams["numIterations"])

    num_col=A.shape[1]

    # Array for residual evolution
    residual1 = []
    residual2 = []
    error_values=[]

    # Create an identity matrix mI
    mIc= sparse.eye(num_col)

    # Calculate mC using Cholesky decomposition
    mC = A.T.dot(A)+lambda_p_1 * (mD.T.dot(mD))+lambda_p_2*sensetive_diag.T.dot(sensetive_diag)+delta_p*mIc 
    # mC = np.linalg.cholesky(prod)
    # mC = cholesky(mI + paramRho * (mD.T @ mD), lower=True)

    # initial conditions
    # Calculate vZ and vU
    vZ = shrink(mD @ vX, alpha / lambda_p_1)
    #vZ = prox_l1(mD @ vX, paramLambda / lambda_p_1)
    vW_ini= shrink(sensetive_diag@vX, beta / lambda_p_2)
    vW=sensetive_diag@vX-vW_ini
    vU = mD @ vX - vZ
    
    y=np.maximum(np.minimum(vX,xMax),xMin)
    w=np.zeros(num_col)


    print('inicio del for')
   
    for ii in tqdm(range(1, numIterations)):
        b = A.T.dot(vY) + lambda_p_1 * mD.T.dot(vZ - vU)+ lambda_p_2 *sensetive_diag.T.dot(vW_ini - vW)+delta_p*(y+w) 
        vX = spsolve(mC, b)
        #vXapr=solver_regu_problem(A,vXapr, vY, paramRhoval,vZ,vU,mD,500100)
        
        # vZ = shrink(mD.dot(vX) + vU/ lambda_p_1, alpha/ lambda_p_1)
        vZ = shrink(mD.dot(vX) + vU, alpha/ lambda_p_1)
        res1 = mD.dot(vX) - vZ
        # vU = vU + lambda_p_1*res1
        vU = vU + res1
        residual1.append(LA.norm(res1))

        # vW_ini = shrink(vX + vW/lambda_p_2, beta/lambda_p_2)
        vW_ini = shrink(sensetive_diag.dot(vX) + vW, beta/lambda_p_2)
        res2 = sensetive_diag.dot(vX) - vW_ini
        # vW = vW + lambda_p_2*res2
        vW = vW + res2

        residual2.append(LA.norm(res2))
        error_values.append(LA.norm(A.dot(vX)-vY))
        w=w+y-vX
        y=np.maximum(np.minimum(vX,xMax),xMin)

        # mX[:, ii] = vX

    # return vX, mX, residual
    return vX, residual1,residual2,vZ,vU,error_values


# def bregman_method_sensitive_1(A,vX, vY, mD,sensetive_diag,alpha,beta,xMax,xMin,sSolverParams):
    
def DG_algorithm_sensitive_2(G,C, V, D,sensetive_diag,alpha,xMax,xMin,sSolverParams):
    
    print('Ususally SB algorithm for any kinfd of structure in norm 1')

    lambda_p = sSolverParams['paramLambda']*sSolverParams['S']
    delta_p = sSolverParams['paramDelta']
    beta_p = sSolverParams['paramBeta']
    numIterations = int(sSolverParams["numIterations"])
    
    # Define the size of vX
    # num_rows = vY.shape[0]
    num_col=G.shape[1]

    # Array for residual evolution
    residual = []
    error_values=[]
    tv_norms=[]
    costs=[]
    d_terms=[]
    box_terms=[]

    # Initialize mX with zeros
    # mX = np.zeros((num_rows, numIterations))
    Id_ec= sparse.eye(num_col)

    # Calculate mC using Cholesky decomposition
    if sSolverParams['lasso']==True:
        mC = G.T.dot(G)+lambda_p * (D.T.dot(D))+(delta_p+beta_p)*Id_ec
    else:
        print('l1 norm')
        mC = G.T.dot(G)+lambda_p * (D.T.dot(D))+delta_p*Id_ec 
    # mC = np.linalg.cholesky(prod)
    # mC = cholesky(mI + paramRho * (mD.T @ mD), lower=True)


    # initial conditions
    # Calculate vZ and vU
    d = shrink(D @ C, alpha / lambda_p)
    #vZ = prox_l1(mD @ vX, paramLambda / lambda_p)
    
    # print('inicio del proceso')
    b = D.dot(C) - d
    
    y=np.maximum(np.minimum(C,xMax),xMin)
    w=np.zeros(num_col)

    # Set the first column of mX to vX
    # mX[:, 0] = vX

    print('inicio del for')
   
    for ii in tqdm(range(1, numIterations)):
        b_rs = G.T.dot(V) + lambda_p * D.T.dot(d - b)+delta_p*(y+w)
        C = spsolve(mC, b_rs)
        #vXapr=solver_regu_problem(A,vXapr, vY, paramRhoval,vZ,vU,mD,500100)
        d = shrink(D.dot(C) + b, alpha/ lambda_p*sensetive_diag)

        #vZ = prox_l1(mD.dot(vX) + vU, paramLambda / paramRhoval)
        res = D.dot(C) - d #difference between gradient
        b = b + res
        residual.append(LA.norm(res))
        
        w=w+y-C
        y=np.maximum(np.minimum(C,xMax),xMin)

        f=LA.norm(G.dot(C)-V)**2
        error_values.append(f)


        tvnorm=LA.norm(d,1)
        tv_norms.append(tvnorm)

        d_term=LA.norm(b+res)**2
        d_terms.append(d_term)

        box_term=LA.norm(C-y-w)**2
        box_terms.append(box_term)



        cost = 0.5*f+ alpha*tvnorm+0.5*lambda_p*d_term+0.5*delta_p*box_term
        costs.append(cost) 
    return C, residual,d,b,error_values,d_terms,box_terms, costs


