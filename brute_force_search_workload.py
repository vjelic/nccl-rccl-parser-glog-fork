#!/usr/bin/python3

import numpy as np # TODO: install check!
import os
import argparse
import csv
import pandas as pd  # TODO: install check!
import time

"""
#                                                       out-of-place                       in-place
#       size         count      type    root     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
    40046416      40046416      int8       0   1268.1   31.58   31.58  0e+00   1279.3   31.30   31.30  0e+00

split_list[-2]
"""
def process_rccl_tests_results(output):
    split_list = output.split("#")
    return split_list[-5].split()[-8] # out-of-place time in microsecond

def generate_permutation():
    permutation_list = []
    default_hip_dev = ','.join(str(i) for i in np.arange(args.gpus_per_node))
    permutation_list.append(default_hip_dev)

    for i in range (0, args.test_iteration):
        arr = np.arange(args.gpus_per_node)
        np.random.shuffle(arr)
        random_hip_dev = ','.join(str(i) for i in arr)
        permutation_list.append(random_hip_dev)
    return permutation_list

def brute_force_search():
    env1 = 'HSA_FORCE_FINE_GRAIN_PCIE=1 '
    env2 = 'HIP_VISIBLE_DEVICES='
    df_net_counts = pd.read_csv(args.net_counts_path, names=["cmd", "count"])
    permutation_list = generate_permutation()
    result_list = []
    start_search = time.time()
    for count, hip_dev in enumerate(permutation_list):
        start = time.time()
        time_elapsed = 0.0
        print("===============================")
        for _, row in df_net_counts.iterrows():
            print(row['cmd'], " ----> ", row['count'])

            cmd = env1 + env2 + hip_dev + ' ' + row['cmd']
            res = os.popen(cmd)
            output = res.read()
            # TODO: if Aborted (core dumped)
            time_in_a_op = float(process_rccl_tests_results(output)) * int(row['count'])
            time_elapsed += time_in_a_op
        end = time.time()
        print("INFO: It takes {} seconds to complete a permutation. Expected time to finish the entire search = {} seconds".format(end - start, (end - start) * (len(permutation_list) - count)))
        print("INFO: For {}, it takes {} seconds for all the RCCL ops.".format(hip_dev, time_elapsed * 1e-6))
        result_list.append([hip_dev, time_elapsed * 1e-6])

    cols = ["HIP_VISIBLE_DEVICES", "Total_time_elapsed (sec)"]
    print("************ It took {} sec for the entire brute-force search. *************".format(time.time() - start_search))
    with open(args.output_csv_name, 'w') as f:
        write = csv.writer(f)
        write.writerow(cols)
        write.writerows(result_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus-per-node', type=int, default=8)
    parser.add_argument('--test-iteration', type=int, default=100, help="Number of randomly-shuffled devie ordering permutations")
    parser.add_argument('--net-counts-path', type=str, default='./net_counts.csv', help="Path to net_counts.csv of a workload")
    parser.add_argument("--output-csv-name", type=str, required=False, default="fambench_cta_2H4P.csv", help="Brute-force search results")
    args = parser.parse_args()
    brute_force_search()

# python brute_force_search_workload.py --gpu-per-node 8 --test-iteration 100 --net_counts_path net_counts.csv --output-csv-name mlperf_resnet50_2H4P.csv
