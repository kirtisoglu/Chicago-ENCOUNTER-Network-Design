import numpy as np
import travel_time as travel
import classes as cl
# -------------------- SA 3: Tabu Search --------------------


# Use tabu list to save visited locations for not producing them again with our random function.


# Short Term Memory: Recency of occurence.

    # Begin with a starting current solution 
    # Create a candidate list of moves.                                          --->  Define candidate lists 
    # Choose the best admissible candidate.                                      --->  Define admissible candidate 
    # If the stopping criterion holds, terminate globally or transfer.           --->  Define stopping criterion                                      
    # Update admissibility conditions. Return 2.                                 --->  Define uptading admissibility condition
    #                                                                          

# Selecting the best admissible candidate

    # (1) Evaluate each candidate move. If there is no move yielding a higher obj value, go to 3.
    # (2) If the move is tabu, check the aspiration level.                                                      --->  Define tabu list 
    #    (2.a) If the move does not satisfy the aspiration criteria, go to 3.                                   --->  Define aspiration criteria 
    #    (2.b) Move is admissible. Designate as the best candidate. Record the admissibility. (why?)                 
    # (3) Candidate list check: If there is no  “good probability” of better moves left,                        
    #     and no reason to extend the candidate list, make the chosen move best admissible, stop.               --->  Define extending candidate list
    # (4) Return 1.  

# Long Term Memory: Frequency of occurence

    # Intensification
    # Diversification

# Definitions
    
    # Candidate list: Blocks in the boundary of the biggest district. 
    # Admissible candidate:  
    # Stopping criterion: It should be a fixed iteration number for now.
    # Updating admissibility condition: No update.
    # Tabu list: Recently visited solutions
    # Tabu Tenure: constant? 
    # Extending candidate list: Pick the next furthest block as a candidate. Extension limit might be tenth furthest block.
    
    # Aspiration Criteria:
    # Extending candidate list:     



 # Questions & Notes

    # Considering all possibilities in a neighborhood might be computationally expensive. Should we remove some of them, maybe randomly? Or pick half of them?
    # Should we change blocks one by one? We could consider two sometimes to make the algorithm faster.
    # This algorithm might be expensive for us in memory usage. Should we record the memory we use? 
    # Which initial solution is the cheapest for calculating in-degrees in boundaries?
    # Admissible candidates for compactness?? in-degree = 3?
    # Which of these operations are costly, which of them reduce the solution space?
    # Is it possible to use diversification without ruining the contiguity?
    # Keep the best solution and restart after each 500 steps?
    # Sometimes unadmissible moves?
    # Admissible moves: Should we assume that the total population must be in a range to change a block?
    
    # Do not forget to update some lists. Tabu list, aspiration list...
    

def tabu_search(initial_solution, max_iterations, tabu_list_size):
	best_solution = initial_solution
	current_solution = initial_solution
	tabu_list = []

	for _ in range(max_iterations):
		neighbors = get_neighbors(current_solution)
		best_neighbor = None
		best_neighbor_fitness = float('inf')

		for neighbor in neighbors:
			if neighbor not in tabu_list:
				neighbor_fitness = objective_function(neighbor)
				if neighbor_fitness < best_neighbor_fitness:
					best_neighbor = neighbor
					best_neighbor_fitness = neighbor_fitness

		if best_neighbor is None:

			# No non-tabu neighbors found,
			# terminate the search
			break

		current_solution = best_neighbor
		tabu_list.append(best_neighbor)
		if len(tabu_list) > tabu_list_size:

			# Remove the oldest entry from the
			# tabu list if it exceeds the size
			tabu_list.pop(0)

		if objective_function(best_neighbor) < objective_function(best_solution):
			# Update the best solution if the
			# current neighbor is better
			best_solution = best_neighbor

	return best_solution


