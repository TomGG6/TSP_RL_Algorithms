import os

from lk_heuristic.utils.solver_funcs import solve


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOLUTION_DIR = os.path.join(BASE_DIR, "temporary_files", "temp-")


def lkh_optimizer(tour, graph, algorithm, runs=1):
    _convert_input(tour, graph, algorithm)
    solve(tsp_file=SOLUTION_DIR + algorithm + ".tsp", solution_method="lk2_improve", runs=runs, backtracking=(5, 5), reduction_level=3,
          reduction_cycle=3, tour_type="cycle", file_name=SOLUTION_DIR + algorithm, logging_level=100)
    output_tour = _convert_output(graph, algorithm)
    return output_tour


def _convert_input(tour, graph, algorithm):
    with open(SOLUTION_DIR + algorithm + ".tsp", "w") as file:
        graph_size = len(graph)
        file.write("NAME : TEMP\n")
        file.write("TYPE : TSP\n")
        file.write("COMMENT : -\n")
        file.write(f"DIMENSION : {graph_size}\n")
        file.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        file.write("NODE_COORD_SECTION\n")
        for i in range(graph_size):
            file.write(f"{i} {graph[tour[i]][0]} {graph[tour[i]][1]}\n")
        file.write("EOF")
        file.close()


def _convert_output(graph, algorithm):
    in_coords_section = False
    tour = []

    with open(SOLUTION_DIR + algorithm + ".tsp", 'r') as file:
        for line in file:
            if line.strip() == "NODE_COORD_SECTION":
                in_coords_section = True
                continue
            if line.strip() in ["EOF", ""]:  # Adjust based on actual end of section marker
                break
            if in_coords_section:
                v = line.split()
                coord = [float(v[1]), float(v[2])]
                tour.append(graph.index(coord))
    tour.append(tour[0])
    return tour
