
def assignment_travel(blocks, all_facilities, locations, travel_time):
    

    locator = {facility: 1 if facility in locations.keys() else 0 for facility in all_facilities.keys()}

    assigner = {(block, facility): 0 for block in blocks.keys() for facility in all_facilities.keys()}

    for block in blocks.keys():
        facility = min((travel_time[(block, facility)], facility) for facility in locations.keys())[1]
        assigner[(block, facility)] = 1 

    return locator, assigner

# For the current block, find the facility with the minimum travel time in existing keys
    # Extract the travel time from the tuple before using the min function
    # The min function selects the tuple with the minimum travel time based on the extracted travel time


def assignment_walk(blocks, all_facilities, locations, walk_time):
    
    locator_walk = {facility: 1 if facility in locations.keys() else 0 for facility in all_facilities.keys()}

    assigner_walk = {(block, facility): 0 for block in blocks.keys() for facility in all_facilities.keys()}

    for block in blocks.keys():
        facility = min((walk_time[(block, facility)], facility) for facility in locations.keys())[1]
        assigner_walk[(block, facility)] = 1 

    return locator_walk, assigner_walk