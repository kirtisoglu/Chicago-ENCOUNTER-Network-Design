import random
import math
import models



def simulated_annealing(initial_solution, max_iterations, temperature, cooling_rate):
    
    current_solution = initial_solution
    current_energy = objective_function(current_solution, N)  # Initial energy (objective function value)
    
    for iteration in range(max_iterations):
        # Generate a neighbor solution by making a small change to the current solution
        neighbor_solution = make_neighbor_solution(current_solution)
        
        # Check constraints for the neighbor solution
        if all(constraint(neighbor_solution) for constraint in constraints):
            # Calculate neighbor energy (objective function value)
            neighbor_energy = objective_function(neighbor_solution, N)
        
            # Calculate energy change (delta E)
            delta_energy = neighbor_energy - current_energy
        
            # If the neighbor solution is better or accepted based on the Metropolis criterion, update the current solution
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
                current_solution = neighbor_solution
                current_energy = neighbor_energy
        
        # Cool the temperature
        temperature *= 1 - cooling_rate
    
    return current_solution, current_energy


def make_neighbor_solution(current_solution): # generates a neighbor solution
    # Implement your logic to generate a neighbor solution (e.g., random changes to variables)
    # List of constraint functions
    constraints = [constraint1, constraint2, constraint3, constraint4, 
                constraint5, constraint6, constraint7, constraint8, constraint9, constraint10]
    pass


def generate_initial_solution():
    
    initial_solution = {}  # Implement your logic to initialize variables here

    return




eps=0.1
def G(X):
    return [(X[0]-2)**2+X[1]**2-1-eps,-(X[0]-2)**2-X[1]**2+1-eps]
    
def RecuitC(Xinit, Tinit, Tf, Ampli, MaxTconst, itermax):
    j=1
    X = Xinit
    T = Tinit
    L =[X]
    Xopt = X
    while(T > Tf and j < itermax):
        compteur = 1
        while(compteur <= MaxTconst):
            Y=[]
            for i in range(len(X)):
                s=X[i]+Ampli*(-1.0+2.0*np.random.random_sample())
                Y.append(s)
            
            # Test if the Y vector satisfies all the constraints
            # if not we choose an another value 

            if G(Y)[0] <=0 and G(Y)[1] <=0: # if G has more two constraints adapt the code    
                DJ=J(Y)-J(X)
                if(DJ < 0.0):
                    X = Y
                    L.append(X)
                    if (J(X) < J(Xopt)):
                        Xopt = X
                else:
                    p = np.random.random_sample()
                    if(p <= np.exp(-DJ / T)):
                        X = Y
                        L.append(X)
            compteur = compteur + 1
        T = g(T)
        j = j + 1
    return [Xopt, L]
# We choose an initial value which satisfies the constraint functions
Xinit = [3,0]
res=RecuitC(Xinit,1000.0,1.0,0.2,2,1500)
print(res[0])
