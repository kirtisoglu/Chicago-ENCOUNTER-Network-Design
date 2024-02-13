
def assignment_travel(grid, stops, blocks, all_facilities, locations, travel_time):
    
    non_facility = {**stops, **blocks}

    locator = {facility: 1 if facility in locations.keys() else 0 for facility in all_facilities.keys()}

    assigner = {(node, facility): 0 for node in grid.keys() for facility in all_facilities.keys()}

    for facility in all_facilities.keys():
        if facility in locations.keys():
            assigner[(facility, facility)] = 1
        else:
            fac = min((travel_time[(facility, facil)], facil) for facil in locations.keys())[1]
            assigner[(facility, fac)] = 1 



    for node in non_facility.keys():
        facility = min((travel_time[(node, facility)], facility) for facility in locations.keys())[1]
        assigner[(node, facility)] = 1 

    return locator, assigner

# For the current block, find the facility with the minimum travel time in existing keys
# Extract the travel time from the tuple before using the min function
# The min function selects the tuple with the minimum travel time based on the extracted travel time


