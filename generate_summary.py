import os
import sys
import argparse
import pandas as pd 

def rccl_tests_log_processor(script_file):
    fs = open(script_file, 'r')
    lines = fs.readlines()
    fs.close()
    results = []
    temp = []
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if ('==========================================================' in line and j != 0):
            results.append(temp)
            temp = []
        elif ("time" in line and "algbw" in line and "busbw" in line):
            perf_line = lines[j+2]
            if ("Avg bus bandwidth" in lines[j + 5]):
                perf_line = perf_line + lines[j + 5]
            elif ("Avg bus bandwidth" in lines[j + 4]):
                perf_line = perf_line + lines[j + 4]
            temp.append(perf_line)
    
    return results

def parse_nccl_performance(perf_lines, net_counts_csv, output_file):
    size, count_1, datatype, op_type, time_oplace, algbw_gbs_oplace, busbw_oplace, error_oplace, time_iplace, algbw_gbs_iplace, busbw_iplace, error_iplace, avg_busbw, count = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    count_table = pd.read_csv(net_counts_csv)
    time_per_op = []
    for i, row in count_table.iterrows():
        group_max_time = float('-inf')
        for line in perf_lines[i]:
            split_list = line.split()
            size.append(split_list[0])
            count_1.append(split_list[1])
            datatype.append(split_list[2])
            op_type.append(split_list[3])
            time_oplace.append(split_list[4])
            algbw_gbs_oplace.append(split_list[5])
            busbw_oplace.append(split_list[6])
            error_oplace.append(split_list[7])
            time_iplace.append(split_list[8])
            algbw_gbs_iplace.append(split_list[9])
            busbw_iplace.append(split_list[10])
            error_iplace.append(split_list[11])
            avg_busbw.append(split_list[-1])
            assert row['count'] % len(perf_lines[i]) == 0
            count.append(int(row['count']//len(perf_lines[i])))
            group_max_time = max(group_max_time, float(split_list[4]) * int(row['count']//len(perf_lines[i])))
        time_per_op.append(group_max_time)
    dict_summary = {"size":size, "count_1":count_1, "datatype":datatype, 
                          "op_type":op_type, "time_oplace":time_oplace, "algbw_gbs_oplace":algbw_gbs_oplace,
                          "busbw_oplace":busbw_oplace, "error_oplace":error_oplace, "time_iplace":time_iplace,
                          "algbw_gbs_iplace":algbw_gbs_iplace, "busbw_iplace":busbw_iplace, "error_iplace":error_iplace,
                          "avg_busbw":avg_busbw, "count":count}
    table = pd.DataFrame(dict_summary)
    table.to_csv(output_file)
    print ("INFO: Dumped out the count of each command in a file named: {}".format(output_file)) 
    return table, time_per_op

def generate_summary_and_time(log_file, count_file, output_file_name):
    log_file = os.path.abspath(log_file)
    count_file = os.path.abspath(count_file)
    out_file = output_file_name + ".csv"
    results = rccl_tests_log_processor(log_file)
    summary_table, time_per_op = parse_nccl_performance(results, count_file, out_file)
    total_time_second = sum(time_per_op)*(1e-6) # microsecond to second
    return total_time_second
    
def main():
    summary_table, total_time = generate_summary_and_time(args.log_file, args.count_file, args.out_file)
    print("Total time spent = {}".format(sum(total_time)*(1e-6))) # microsecond to second
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True, help="Log file generated while running rccl-tests with net_unique_topo.sh")
    parser.add_argument("--output-file-name", type=str, required=False, default="net_summary")
    parser.add_argument("--count-file", type=str, required=False, default="net_counts.csv", help="net_count file generated while running unique option in parser.")

    args = parser.parse_args()
    main()

# python generate_summary.py --log-file MP_first_MP_rccl_tests.txt --count-file net_counts_MP.csv --output-file-name MP_first_MP 