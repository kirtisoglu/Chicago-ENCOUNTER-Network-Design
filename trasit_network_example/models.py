import classes as cl


max_cost = 100
traveltime_threshold = 30
max_DN_teams = 4
max_new_PHCs = 10
alpha = 0.02


"-------Model 1: Integer Programming Model-------"

# Decision Variables

             # y_j ∈ {0, 1} locates facility j.

def objective_function():  # Accessibility function
    return

def constraint_1(E, y_j): 
    for all j in E
    y_j = 1
    return



# Creating the task_schedule with the correct dimensions (num_groups_corrected, num_tasks, num_days)
task_schedule = np.zeros((num_groups_corrected, num_tasks, num_days), dtype=int)

def constraint_1(task_schedule):
    total_overlaps = 0
    
    # Iterate through days and tasks to count overlaps
    for day in range(num_days):
        for task in range(num_tasks):
            groups_assigned = np.where(task_schedule[:, task, day] == 1)[0]
            overlaps = len(groups_assigned) - 1 
            total_overlaps += overlaps
    
    return total_overlaps

def constraint_2(task_schedule):
    total_double_assignments = 0
    
    # Iterate through days and groups to count double assignments
    for day in range(num_days):
        for group in range(num_groups_corrected):
            tasks_assigned = np.where(task_schedule[group, :, day] == 1)[0]
            assignments = len(tasks_assigned)
            if assignments > 1:
                total_double_assignments += assignments - 1
            else:
                total_double_assignments=0
    return total_double_assignments

def constraint_3(task_schedule):
    total_overlaps_of_task = 0
    
    for group in range(num_groups_corrected):
        for task in range(num_tasks):
            task_assigned = np.where(task_schedule[group, task, : ] == 1)[0]
            overlaps = len(task_assigned) - 1 
            total_overlaps_of_task += overlaps
            if total_overlaps_of_task > 1:
                return total_overlaps_of_task
            else:
                return 0

"-------Model 2: Block-based Model-------"


"-------Model 3: District-based Model-------"


"-------Model 4: Huff-based Model-------"