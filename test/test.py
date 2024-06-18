import os
from multiprocessing import Process

import pandas as pd

from ant_colony_algorithm.run import run_ant_colony_algorithm
from genetic_algorithm.genetic_algorithm import run_genetic_algorithm
from lkh.run import run_lkh_algorithm
from vsr_lkh.run import run_vsr_lkh_algorithm, run_vsr_expected_lkh_algorithm

DATA = {
    "berlin52.txt": 7542,
    "pr76.txt": 108159.0,
    "pr144.txt": 58537,
    "kroB150.txt": 26130,
    "pr226.txt": 80369,
    "pr299.txt": 48191,
    "pcb442.txt": 50778,
    "d493.txt": 35002,
    "si535.txt": 48450,
    "rat575.txt": 6773,
    "d657.txt": 48912,
    "pr1002.txt": 259045,
    "u1060.txt": 224094,
    "vm1084.txt": 239297,
    "pcb1173.txt": 56892,
    "rl1304.txt": 252948,
    "rl1323.txt": 270199,
    "nrw1379.txt": 56638,
    "vm1748.txt": 336556,
    "u1817.txt": 57201,
    "rl1889.txt": 316536,
    "d2103.txt": 80450,
    "u2152.txt": 64253,
    "pcb3038.txt": 137694,
    "fnl4461.txt": 182566,
    "rl5915.txt": 565530,
    "rl5934.txt": 556045,
}

ALGORITHMS = {
    "GA": run_genetic_algorithm,
    "AC": run_ant_colony_algorithm,
    "LKH": run_lkh_algorithm,
    "Q-LKH": run_vsr_lkh_algorithm,
    "SARSA-LKH": run_vsr_lkh_algorithm,
    "EXPECTED-SARSA": run_vsr_lkh_algorithm,
    "MONTECARLO-LKH": run_vsr_lkh_algorithm,
    "TD-LKH": run_vsr_lkh_algorithm,
    "VSR-LKH": run_vsr_lkh_algorithm,
    "VSR-V2-LKH": run_vsr_lkh_algorithm,
    "VSR-EXPECTED-LKH": run_vsr_expected_lkh_algorithm
}


def run_test(algorithm, num_test):
    filename = "results" + algorithm + ".csv"
    columns = ["ALGORITHM", "FILE", "OPTIMUM", "AVERAGE", "BEST", "WORST", "SUCCESS", "LOSS", "TIME"]

    file_exists = os.path.exists(filename)
    for file in DATA:
        results = []
        for sample in range(num_test):
            if algorithm == "Q-LKH":
                cost, time = ALGORITHMS[algorithm]("data/" + file, M=1, is_vsr=False, optimum=DATA[file], algorithm=algorithm)
            elif algorithm == "SARSA-LKH":
                cost, time = ALGORITHMS[algorithm]("data/" + file, M=2, is_vsr=False, optimum=DATA[file], algorithm=algorithm)
            elif algorithm == "MONTECARLO-LKH":
                cost, time = ALGORITHMS[algorithm]("data/" + file, M=3, is_vsr=False, optimum=DATA[file], algorithm=algorithm)
            elif algorithm == "TD-LKH":
                cost, time = ALGORITHMS[algorithm]("data/" + file, M=4, is_vsr=False, optimum=DATA[file], algorithm=algorithm)
            elif algorithm == "EXPECTED-SARSA":
                cost, time = ALGORITHMS[algorithm]("data/" + file, M=5, is_vsr=False, optimum=DATA[file], algorithm=algorithm)
            elif algorithm == "VSR-LKH":
                cost, time = ALGORITHMS[algorithm]("data/" + file, optimum=DATA[file], algorithm=algorithm)
            elif algorithm == "VSR-V2-LKH":
                cost, time = ALGORITHMS[algorithm]("data/" + file, num_M=5, optimum=DATA[file], algorithm=algorithm)
            elif algorithm == "VSR-EXPECTED-LKH":
                cost, time = ALGORITHMS[algorithm]("data/" + file, optimum=DATA[file], algorithm=algorithm) 
            else:
                cost, time = ALGORITHMS[algorithm]("data/" + file)
            results.append([cost, time])
            print(f"file: {file}, sample: {sample}, results: {[cost, time]}")
        average = sum([i[0] for i in results]) / num_test
        total_time = sum([i[1] for i in results]) / num_test
        best = min([i[0] for i in results])
        worst = max([i[0] for i in results])
        success = sum(1 for result in results if result[0] <= DATA[file])
        loss = (average - DATA[file]) / DATA[file] * 100

        row_df = pd.DataFrame([{
            "ALGORITHM": algorithm,
            "FILE": file,
            "OPTIMUM": DATA[file],
            "AVERAGE": average,
            "BEST": best,
            "WORST": worst,
            "SUCCESS": f"{success}/{num_test}",
            "LOSS": f"{loss}%",
            "TIME": f"{total_time}s",
        }], columns=columns)

        row_df.to_csv(filename, mode='a', header=not file_exists, index=False)
        file_exists = True


def run_tests(num_test):
    processes = []
    for algorithm in ALGORITHMS:
        p = Process(target=run_test, args=(algorithm, num_test))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    run_tests(5)
