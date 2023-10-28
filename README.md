# Chicago-ENCOUNTER-Network-Design

### Edge Class

Information related to all arcs. Contains the following columns:

**-ID:** Unique identifying number.
**-Type:** Arc type ID. The types in use are:
	0: line arc
	1: boarding arc
	2: alighting arc
	3: core network walking arc (stop nodes only)
	4: accessibility network walking arc (stop nodes, population centers, and facilities)
**-Line:** Line ID of a line arc, and -1 otherwise.
**-Tail:** Node ID of the arc's tail.
**-Head:** Node ID of the arc's head.
**-Travel Time:** Constant part of travel time of arc. Boarding arcs, whose travel time is based on the line frequency, have a listed time of 0.
