"1. Set the initial step counter to 0, set the initial temperature T, and choose an initial solution x ∈ X."
import models 

tempature = 0.99
step_counter = 0

def generate_initial_solution():
    return

"2. Choose a neighbor y ∈ N(x)."

def generate_neighborhood():
    return


def compare_objective_values():
    return

"(a) If f(y) ≤ f(x), then make y our candidate solution and proceed to Step 3."


"(b) Otherwise, calculate and apply the acceptance probability P(x, y, T)." 
"If we choose to accept, then y is our candidate and we proceed to Step 3."
"If not, we choose a different neighbor of x and return to Step 2."


"3. Set x ← y, making the candidate solution our new current solution. Increment the step counter." 
"Apply the temperature schedule to T."

"4. Determine if we have reached the stopping criterion."

"(a) If yes, then stop and return the best solution found so far."

"(b) If no, then go back to Step 2."