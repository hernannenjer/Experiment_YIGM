import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import plotly
from mpl_toolkits.mplot3d import Axes3D
from numpy import random
from numpy import linalg as LA
from scipy.linalg import svd
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

# Import utils for graphs
import matplotlib.pyplot as plt

# Misc
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

def signal(x, s, mu,dV_x):
    ex=np.array([1.0,0.0,0.0])
    rk=x-s
    nrk=np.linalg.norm(rk)
    return (mu/(4*np.pi))*(3*rk*(np.dot(rk,ex))/(nrk**5)-ex/(nrk**3))*dV_x


def Gsignal(x, s, d_vector,dV_x,mu):
    #dV_x will be constant for every subvolumns
    #d_vector: associated with the directions of the position of sensor
    #x : position of the vector of the sources
    #s : positions of sensors

    e = d_vector / np.linalg.norm(d_vector) #unitary vector e
    ex=np.array([1.0,0.0,0.0])
    rk=x-s
    nrk=np.linalg.norm(rk)
    return (mu/(4*np.pi))*((3*(np.dot(rk,e))*(np.dot(rk,ex)))/(nrk**5)-(np.dot(ex,e)/(nrk**3)))*dV_x


def Gmatrixsignal(x, s, d_vector,dV_x,mu):
    #for the moment dV_x will be constant
    return np.array([[Gsignal(x[j],s[k],d_vector,dV_x,mu) for j in range(len(x))] for k in range(len(s))])

#######################################################################################
#put the vector in the direction of taking signal e for the magnometer direcction
######################################################################################

def Gmatrixsignal_sensor(x, s,dV_x,mu):
    #d_vector=s[k]
    return np.array([[Gsignal(x[j],s[k],s[k],dV_x,mu) for j in range(len(x))] for k in range(len(s))])


def Gsignal1(x, s, d_vector, dV_x, mu):
    e = d_vector / np.linalg.norm(d_vector)  # Precompute unitary vector e
    ex = np.array([1.0, 0.0, 0.0])
    rk = x - s
    nrk = LA.norm(rk)
    nrk3 = nrk ** 3
    nrk5 = nrk ** 5
    dot_rk_e = np.dot(rk, e)
    dot_rk_ex = np.dot(rk, ex)
    dot_ex_e = np.dot(ex, e)
    return (mu / (4 * np.pi)) * ((3 * dot_rk_e * dot_rk_ex) / nrk5 - (dot_ex_e / nrk3)) * dV_x


def Gmatrixsignal_sensor1(xs, ss, dV_x, mu):
    return np.array([[Gsignal1(x, s, s, dV_x, mu) for x in xs] for s in ss])

def sigmaF(Mn, Mb, Mr, Tn, Tb,t):
    return Mn*np.exp(-t/Tn)+Mb*np.exp(-t/Tb)+Mr


def gradient_x_matrix(numx,numy,numz):

    mT = diags([np.ones(numx), -np.ones(numx)], [0, 1], shape=(numx-1, numx))

    # sabiendo que la concentracion fuera del dominio es cero
    #mT = diags([np.hstack((np.ones(numx-1),0)), -np.ones(numx)], [0, 1], shape=(numx, numx))
    mDv = kron(mT, np.eye(numy))
    mDx = kron(mDv,np.eye(numz))

    #mD = vstack([mDv, mDh], format='csc')  # Use 'csc' format for better sparse matrix performance

    return mDx


def gradient_y_matrix(numx,numy,numz):

    mT = diags([np.ones(numy), -np.ones(numy)], [0, 1], shape=(numy-1, numy))

    # sabiendo que la concentracion fuera del dominio es cero
    #mT = diags([np.hstack((np.ones(numy-1),0)), -np.ones(numy)], [0, 1], shape=(numy, numy))
    mDv = kron(mT, np.eye(numz))
    mDy = kron(np.eye(numx), mDv)

    #mD = vstack([mDv, mDh], format='csc')  # Use 'csc' format for better sparse matrix performance

    return mDy

def gradient_z_matrix(numx,numy,numz):
    # Vertical Operator - T(numRows)
    mT = diags([np.ones(numz - 1), -np.ones(numz - 1)], [0, 1], shape=(numz - 1, numz))
    #mT = diags([np.hstack((np.ones(numz -1),0)), -np.ones(numz-1 )], [0, 1], shape=(numz , numz))
    mDv = kron(np.eye(numy), mT)
    mDz = kron(np.eye(numx), mDv)

    return mDz


#################
#general gradient

def create_gradient_operator(numx, numy, numz):
    # Vertical Operator - T(numRows)

    mD = vstack([gradient_x_matrix(numx,numy,numz) ,gradient_y_matrix(numx,numy,numz), gradient_z_matrix(numx,numy,numz)], format='csc')
    # Use 'csc' format for better sparse matrix performance

    return mD

def shrink(a, delta):
    return np.sign(a) * np.maximum(np.abs(a) - delta, 0)

# Functions for ADMM steps

def prox_l1(vX, lambdaFactor):
    # Soft Thresholding
    vX = np.maximum(vX - lambdaFactor, 0) + np.minimum(vX + lambdaFactor, 0)
    return vX


def SB_TV_v3(A,vX, vY, mD,paramLambda,sSolverParams):
    paramRhoval = sSolverParams['paramRho']
    numIterations = int(sSolverParams["numIterations"])
    # Define the size of vX
    num_rows = vY.shape[0]

    # Array for residual evolution
    residual = []

    # Initialize mX with zeros
    # mX = np.zeros((num_rows, numIterations))

    # Create an identity matrix mI
    mI = sparse.eye(num_rows)

    # Calculate mC using Cholesky decomposition
    mC = A.T@A+paramRhoval * (mD.T @ mD)  #mI + paramRho * (mD.T @ mD)
    # mC = np.linalg.cholesky(prod)
    # mC = cholesky(mI + paramRho * (mD.T @ mD), lower=True)


    # initial conditions
    # Calculate vZ and vU
    vZ = shrink(mD @ vX, paramLambda / paramRhoval)
    #vZ = prox_l1(mD @ vX, paramLambda / paramRhoval)
    
    
    vU = mD @ vX - vZ

    # Set the first column of mX to vX
    # mX[:, 0] = vX
    vXapr=vX
    for ii in tqdm(range(1, numIterations)):
        b = A.T@vY + paramRhoval * mD.T.dot(vZ - vU)
        vX = spsolve(mC, b)
        #vXapr=solver_regu_problem(A,vXapr, vY, paramRhoval,vZ,vU,mD,500100)
        
        
        
        vZ = shrink(mD.dot(vX) + vU, paramLambda / paramRhoval)
        #vZ = prox_l1(mD.dot(vX) + vU, paramLambda / paramRhoval)
        res = mD.dot(vX) - vZ
        vU = vU + res
        residual.append(LA.norm(res))

        # mX[:, ii] = vX

    # return vX, mX, residual
    return vX, residual,vZ,vU



def SB_TV_conjugated_gradient(A,vX, vY, mD,alpha,xMax,XMin,sSolverParams,max_its=100):
    #with constrains 
    lambda_p = sSolverParams['paramLambda']
    numIterations = int(sSolverParams["numIterations"])
    # Define the size of vX
    num_col=A.shape[1]
    lower = XMin*np.ones(num_col)
    upper = xMax*np.ones(num_col)
    # Array for residual evolution
    residual = []
    # calculate the matrix for solving the minimization quadratic constrain problem 
    mCC = A.T.dot(A) + lambda_p * (mD.T.dot(mD))

    # initial conditions
    # Calculate vZ and vU
    vZ = shrink(mD @ vX, alpha / lambda_p)
    vU = mD @ vX - vZ
    
    # y=np.maximum(np.minimum(vX,xMax),XMin)
    # w=np.zeros(num_col)

    # Set the first column of mX to vX
    # mX[:, 0] = vX
   
    for ii in tqdm(range(1, numIterations)):
        bb = A.T.dot(vY) + lambda_p * mD.T.dot(vZ - vU)
        solver = GPCGSolver(mCC, bb, lower, upper)
        vX = solver.solve(maxits=max_its)['x']
        # vX = spsolve(mC, b)
        #vXapr=solver_regu_problem(A,vXapr, vY, paramRhoval,vZ,vU,mD,500100)
        vZ = shrink(mD.dot(vX) + vU, alpha/ lambda_p)
        #vZ = prox_l1(mD.dot(vX) + vU, paramLambda / paramRhoval)
        res = mD.dot(vX) - vZ
        vU = vU + res
        residual.append(LA.norm(res))
    return vX, residual,vZ,vU




def SB_TV_pos_const(A,vX, vY, mD,alpha,xMax,XMin,sSolverParams):
    #with constrains 
    lambda_p = sSolverParams['paramLambda']
    delta_p = sSolverParams['paramDelta']
    numIterations = int(sSolverParams["numIterations"])
    # Define the size of vX
    num_rows = vY.shape[0]
    num_col=A.shape[1]

    # Array for residual evolution
    residual = []

    # Initialize mX with zeros
    # mX = np.zeros((num_rows, numIterations))

    # Create an identity matrix mI
    mI = sparse.eye(num_rows)
    mIc= sparse.eye(num_col)

    # Calculate mC using Cholesky decomposition
    mC = A.T.dot(A)+lambda_p * (mD.T.dot(mD))+delta_p*mIc 
    # initial conditions
    # Calculate vZ and vU
    vZ = shrink(mD @ vX, alpha / lambda_p)
    vU = mD @ vX - vZ
    
    y=np.maximum(np.minimum(vX,xMax),XMin)
    w=np.zeros(num_col)

    # Set the first column of mX to vX
    # mX[:, 0] = vX
   
    for ii in tqdm(range(1, numIterations)):
        b = A.T.dot(vY) + lambda_p * mD.T.dot(vZ - vU)+delta_p*(y+w)
        vX = spsolve(mC, b)
        #vXapr=solver_regu_problem(A,vXapr, vY, paramRhoval,vZ,vU,mD,500100)
        vZ = shrink(mD.dot(vX) + vU, alpha/ lambda_p)
        #vZ = prox_l1(mD.dot(vX) + vU, paramLambda / paramRhoval)
        res = mD.dot(vX) - vZ
        vU = vU + res
        residual.append(LA.norm(res))
        w=w+y-vX
        y=np.maximum(np.minimum(vX,xMax),XMin)
        # mX[:, ii] = vX
    return vX, residual,vZ,vU




def sol_inv_prob_num_mea(G,cj,n,t_range,excitations,Mn,Mb,Mr,Tn,Tb, xMax=30,xMin=0, n_iterations=500):
    results_01={} #dictionary for store the results
    for n_excita in excitations:
        #consider the  time when the coils where turn off 
        t_measurements=np.random.choice(t_range,size=n_excita,replace=True)
        print(len(t_measurements))
        #data recording by the sensors with noise
        Vdata_cA=np.concatenate([sigmaF(Mn, Mb, Mr, Tn, Tb,t)*G.dot(cj) for t in t_measurements], axis=0)
        noise_cA=np.random.normal(0,1,Vdata_cA.shape[0])
        epsilon_A=LA.norm(Vdata_cA)/(10*LA.norm(noise_cA))
        Vdata_noise_cA=Vdata_cA+epsilon_A
        #matrix lead field 
        A=np.concatenate([sigmaF(Mn, Mb, Mr, Tn, Tb,t)*G for t in t_measurements],axis=0)
        #matrix for the total variotion
        mD=create_gradient_operator(n-1,n-1 ,n-1)
        #for solving the TV norm
        uinit = np.ones((n-1)*(n-1)*(n-1))
        alpha=1e-13
        sSolverParams={
            'paramLambda': 1e-13,
            'paramDelta': 1e-7,
            'numIterations': n_iterations
            }
    
        u_A, residual, zk,b=SB_TV_pos_const(A,uinit, Vdata_noise_cA, mD,alpha,xMax,xMin,sSolverParams)
    
        # Create the variable name dynamically
        var_name = "u_01_" + str(n_excita)
    
        # Store the result in the results dictionary
        results_01[var_name] = {'u_A': u_A, 'residual': residual, 'zk': zk, 'b': b}
    return results_01


def grid_volume(cj,h,domain):
  ax,bx=domain['x']

  ay,by=domain['y']

  az,bz=domain['z']
  #Fucion para graficar

  nx=int((bx-ax)/h)+1
  ny=int((by-ay)/h)+1
  nz=int((bz-az)/h)+1

  dx = (bx-(ax))/(nx-1)
  dy = (by-(ay))/(ny-1)
  dz = (bz-(az))/(nz-1)

  matrixcjexa = np.reshape(cj, (nx-1, ny-1, nz-1))

  values = matrixcjexa
  values.shape

  # Create the spatial reference
  grid = pv.ImageData()

  # Set the grid dimensions: shape + 1 because we want to inject our values on
  #   the CELL data
  grid.dimensions = np.array(values.shape)+1

  # Edit the spatial reference
  grid.origin = (ax,ay,az)  # The bottom left corner of the data set
  grid.spacing = (dx, dy, dz)  # These are the cell sizes along each axis

  # Add the data values to the cell data
  grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array
  
  return grid



def plot_volume_sensors(grid,s=None,axis=True,nb=True):
    #Fucion para graficar
    plotter = pv.Plotter(notebook=nb)
    plotter.add_mesh(grid,show_edges=True, opacity=0.9)
    if s is not None:
        for l in range(len(s)):
            plotter.add_mesh(s[l],opacity=1, color='red',render_points_as_spheres=True, point_size=8)
    if axis==True:
        plotter.show_axes()
    plotter.show()



def domain_mesh(h,domain,cotas):
    '''This function returns the coordinates of the doamin discretized by finite differences in size h, 
    the volume of each voxel and the mask of  the cotas of  cube of cuadricula form volume the volume that contains the concentration of the MN particles'''
    
    ax,bx=domain['x']

    ay,by=domain['y']

    az,bz=domain['z']


    nx=int((bx-ax)/h)+1
    ny=int((by-ay)/h)+1
    nz=int((bz-az)/h)+1
    dx = (bx-(ax))/(nx-1)
    dy = (by-(ay))/(ny-1)
    dz = (bz-(az))/(nz-1)


    x = np.linspace(ax+dx/2, bx-dx/2, nx-1)
    y = np.linspace(ay+dy/2, by-dy/2, ny-1)
    z = np.linspace(az+dz/2, bz-dz/2, nz-1)

    ###source positions
    #coordinates1 = np.array([(a, b, c)  for a in x for b in y for c in z])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    coordinates = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    nc=len(coordinates)
    #volume for every single volume
    V=(bx-ax)*(by-ay)*(bz-az)
    
    dV_x=V/nc


    eps = 1e-10 


    #mask by using cotas

    if cotas['cube']==False:
        radius=cotas['radius']
        center=np.array(cotas['center'])
        mask= (LA.norm((coordinates[:]-center),axis=1)<=radius+eps)
        

    else:
        x1,x2=cotas['x']
        y1,y2=cotas['y']
        z1,z2=cotas['z']
        mask= (((coordinates[:,0]+dx/2<=x2+eps)&(x1-eps<=coordinates[:,0]-dx/2))  & ((coordinates[:,1]+dy/2<=y2+eps)&(y1-eps<=coordinates[:,1]-dy/2)) & (coordinates[:,2]+dz/2<=z2+eps) & (z1-eps<=coordinates[:,2]-dz/2))

    return mask,coordinates,dV_x,nx,ny,nz


def iteration_foward_model(s,h_values,domain,cotas,b_real,psi_t, concentration_obj,mu):
    '''create a loop for different h values of the domain, and get the forward model for a sensor set  in a domain with a concentration values expressed in cotas'''
    Vdata_1=[]
    norm_Vdata_1=[]
    for h in tqdm(h_values):
        mask1,coordinates1,dV_x,_,_,_=domain_mesh(h,domain,cotas)
        cj = np.zeros(len(coordinates1))
        cj[mask1]=concentration_obj
        G=Gmatrixsignal_sensor1(coordinates1, s,dV_x,mu)
        b_approx=np.dot(psi_t*G,cj)
        Vdata_1.append(b_approx)
        norm_Vdata_1.append(LA.norm(b_approx-b_real))
    print(norm_Vdata_1)
    return Vdata_1,norm_Vdata_1


def kernel(mass_center,s, mu,ex):
    values=[]
    for si in s:
        unitario=si/LA.norm(si) 
        r_c=mass_center-si
        nr_c=LA.norm(r_c)
        value_vector=mu/(4*np.pi)*(3*ex.dot(r_c)*r_c/nr_c**5-ex/nr_c**3)
        values.append(value_vector.dot(unitario))
    return np.array(values)

def center_mass(cotas):
    x1,x2=cotas['x']
    y1,y2=cotas['y']
    z1,z2=cotas['z']
    mass_center=np.array([x2+x1,y2+y1,z2+z1])/2
    volume_voxel=(x2-x1)*(y2-y1)*(z2-z1)
    return mass_center,volume_voxel


def get_coil_points_axis(num_segments_per_turn, helmet_center, a, b, axis):
    points = []
    for segment in range(num_segments_per_turn):
        theta_start = segment * 2 * np.pi / (num_segments_per_turn)
        theta_end = (segment + 1) * 2 * np.pi / (num_segments_per_turn)
        
        if axis == 'y':
            x1 = helmet_center[0] + a * np.cos(theta_start)
            y1 = helmet_center[1]
            z1 = helmet_center[2] + b * np.sin(theta_start)
            x2 = helmet_center[0] + a * np.cos(theta_end)
            y2 = helmet_center[1]
            z2 = helmet_center[2] + b * np.sin(theta_end)
        elif axis == 'x':
            x1 = helmet_center[0]
            y1 = helmet_center[1] + a * np.cos(theta_start)
            z1 = helmet_center[2] + b * np.sin(theta_start)
            x2 = helmet_center[0]
            y2 = helmet_center[1] + a * np.cos(theta_end)
            z2 = helmet_center[2] + b * np.sin(theta_end)
        elif axis == 'z':
            x1 = helmet_center[0] + a * np.cos(theta_start)
            y1 = helmet_center[1] + b * np.sin(theta_start)
            z1 = helmet_center[2]
            x2 = helmet_center[0] + a * np.cos(theta_end)
            y2 = helmet_center[1] + b * np.sin(theta_end)
            z2 = helmet_center[2]
        else:
            raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")
        
        points.append([(x1, y1, z1), (x2, y2, z2)])
    return points


def coil_magnetic_field(rv, coil_segments, coil_current):
    """
    Calculate the magnetic field at position rv due to a coil with multiple line segments.
    
    Parameters:
    rv (numpy.ndarray): Position vector of the vth voxel.
    coil_segments (numpy.ndarray): Array of start and end points of the coil line segments.
    coil_current (float): Current carried by the coil.
    
    Returns:
    numpy.ndarray: Magnetic field at position rv.
    """
    Hc_v = 0
    for f in coil_segments:
        f1=np.array(f[0])
        f2=np.array(f[1])
        r1 = rv - f1
        r2 = rv - f2
        
        norm_r1 = np.linalg.norm(r1)
        norm_r2 = np.linalg.norm(r2)
        
        f1_cross_f2 = np.cross(r1, r2)
        norm_f1 = np.linalg.norm(r1)
        norm_f2 = np.linalg.norm(r2)
        dot_f1_f2 = np.dot(r1, r2)
        
        Hc_v += (norm_r1 + norm_r2) * f1_cross_f2 * coil_current / (norm_f1 * norm_f2 * (norm_f1 * norm_f2 + dot_f1_f2))
    
    Hc_v /= (4 * np.pi)
    
    return Hc_v

def lead_field_matrix_coil(sources, sensors, a_s, coil_segments, coil_current):
    # Extract the coordinates of the sources and sensors
    x = sensors
    xi = sources      

    # Compute the distance vector between sources and sensors
    r_s = x[:, None] - xi[None, :]
    r = np.linalg.norm(r_s, axis=2)
    # Compute the unit vector in the x-direction where is located the coil

    # e_x = np.array([1, 0, 0])


    Hc_v = np.zeros_like(xi)
    for f in coil_segments:
        f1=np.array(f[0])
        f2=np.array(f[1])
        r1 = np.squeeze(xi[None, :] - f1)
        r2 = np.squeeze(xi[None, :] - f2)
        norm_f1 = np.linalg.norm(r1,axis=1)
        norm_f2 = np.linalg.norm(r2,axis=1)
        f1_mult_wis_f2=np.multiply(norm_f1 , norm_f2)
        f1_cross_f2 = np.cross(r1, r2, axis=1)
        dot_f1_f2 = np.einsum('ij,ij->i', r1, r2)
        numerador=(norm_f1 + norm_f2)* coil_current
        denominador=np.multiply(f1_mult_wis_f2, (f1_mult_wis_f2 + dot_f1_f2))
    
        Hc_v +=  np.divide(numerador,denominador)[:,None]*(f1_cross_f2)
    Hc_v /= (4 * np.pi)



    # Compute the unit vector in the direction of the sensor
    norm_sensors=np.linalg.norm(a_s, axis=1)
    e_s = a_s/ norm_sensors[:,None]

    # Compute the matrix elements
    first_term=3/(r ** 5)
    first_term = np.array([[first_term[s,i]* (r_s[s,i,:].dot(e_s[s])) * (Hc_v[i].dot(r_s[s,i,:]))for i in range(len(xi)) ]  for s in range(len(x))])
    second_term = 1 / (r ** 3)
    second_term = np.array([[second_term[s,i]*  (Hc_v[i].dot(e_s[s]))for i in range(len(xi)) ]  for s in range(len(x))])
    G = first_term - second_term
    print('G.shape = '+str(G.shape))

    return G

def regularization_for_l1(G, uinit, V, alpha, xMax, xMin, parameters):
    residual = []
    relative_diff = []
    # beta = 0.9 * LA.norm(G.T.dot(G))
    # print(beta)
    beta=1e2
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