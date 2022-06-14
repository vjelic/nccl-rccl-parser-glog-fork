import os
import sys
import csv
import argparse

coll_op_map = {
            "Broadcast": "broadcast_perf",
            "Reduce": "reduce_perf",
            "AllGather": "all_gather_perf",
            "ReduceScatter": "reduce_scatter_perf",
            "AllReduce": "all_reduce_perf",
            "Gather": "gather_perf",
            "Scatter": "scatter_perf",
            "AllToAll": "alltoall_perf",
            "AllToAllv": "alltoallv_perf",
            "Send": "sendrecv_perf",
            "Recv": "sendrecv_perf",
          }

reduction_op_map = {
                "0" : "sum",
                "1" : "prod",
                "2" : "max",
                "3" : "min",
                "4" : "all",
               }

data_types_map = {
                "0" : "int8",
                "1" : "uint8",
                "2" : "int32",
                "3" : "uint32",
                "4" : "int64",
                "5" : "uint64",
                "6" : "half",
                "7" : "float",
                "8" : "double",
                "9" : "bf16",
                #"10" : "ncclNumTypes Equivalent?"
             }

data_type_bytes_map = {
                    "0" : 1,
                    "1" : 1,
                    "2" : 4,
                    "3" : 4,
                    "4" : 8,
                    "5" : 8,
                    "6" : 2,
                    "7" : 4,
                    "8" : 8,
                    "9" : 2,
                    #"10" : Not sure.
                  }

def get_useful_info(log_file):
    fs = open(log_file, 'r')
    lines = fs.readlines()
    fs.close()

    useful_lines = []
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if ("opCount" in line and "sendbuff" in line):
            useful_lines.append(line)

    return useful_lines

def parse_nccl_log(nccl_lines):

    commands = []
    for j in range(len(nccl_lines)):
        line = nccl_lines[j]
        split_list = line.split(" ")
        comm = split_list[split_list.index("INFO") + 1].replace(":", "")
        count = split_list[split_list.index("count") + 1]
        datatype = split_list[split_list.index("datatype") + 1]
        op_type = split_list[split_list.index("op") + 1]
        root = split_list[split_list.index("root") + 1]
        nnranks = next(item for item in split_list if 'nranks' in item).split("=")[1].replace("]", "")

        #print (comm)
        #print (count)
        #print (datatype)
        #print (op_type)
        #print (root)
        #print (nnranks)

        total_bytes = int(count) * data_type_bytes_map[datatype]

        test_cmd = "./build/" + coll_op_map[comm] + " -d " + data_types_map[datatype] + \
                       " -b " + str(total_bytes) + " -e " + str(total_bytes) + \
                       " -o " + reduction_op_map[op_type] + " -g " + str(nnranks)
        #print (test_cmd)
        commands.append((test_cmd, int(nnranks)))

    return commands

def generate_script(commands, output_script):
    filename = output_script + ".sh"
    fs = open(filename, "w")
    for j in range(len(commands)):
        fs.write(commands[j])
        fs.write("\n")
    fs.close()
    print("INFO: Dumped out the commands in a script named: {}".format(filename))

def dump_counts_map(counts_map, output_file):
    filename = output_file + ".csv"
    result_list = []
    keys = counts_map.keys()
    for key in keys:
        result_list.append([key, str(counts_map[key])])
    
    with open(filename, 'w') as f:
        write = csv.writer(f)
        write.writerows(result_list)

    print ("INFO: Dumped out the count of each command in a csv file named: {}".format(filename))

def get_unique_commands(commands_and_nranks):
    unique_values = []
    counts_map = {}
    nranks_map = {}
    for c_and_nr in commands_and_nranks:
        cmd = c_and_nr[0]
        nranks = c_and_nr[1]
        if (cmd not in unique_values):
            counts_map[cmd] = 1
            nranks_map[cmd] = nranks
            unique_values.append(cmd)
        else:
            counts_map[cmd] = counts_map[cmd] + 1
    assert len(counts_map) == len(nranks_map)
    for cmd in counts_map.keys():
        # (TODO) It's weird that some ops with nranks = 8 only have 7 collective line in a MLPerf ResNet50 log....
        #print(cmd, " === ", nranks_map[cmd], " === ", counts_map[cmd])
        #assert counts_map[cmd] % nranks_map[cmd] == 0
        counts_map[cmd] = int(counts_map[cmd] / nranks_map[cmd])
    return unique_values, counts_map

def main():
    log_file = os.path.abspath(args.nccl_debug_log)
    nccl_lines = get_useful_info(log_file)
    commands_and_nranks = parse_nccl_log(nccl_lines)
    #generate_script(commands, args.output_script_name)
    if (args.unique):
        new_commands, counts_map = get_unique_commands(commands_and_nranks)
        generate_script(new_commands, args.output_script_name + "_unique")
        dump_counts_map(counts_map, args.output_script_name + "_counts")
    else:
        commands = list(zip(*commands_and_nranks))[0]
        generate_script(commands, args.output_script_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nccl-debug-log", type=str, required=True, help="Log from app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL")
    parser.add_argument("--output-script-name", type=str, required=False, default="net_nccl_rccl", help="Output command script")
    parser.add_argument("--unique", action="store_true", default=False, help="Get only the unique commands.")
    parser.add_argument("--tuning", action="store_true", default=False, help="Use brute force search for HIP_VISIBLE_DEVICES ordering permuntations.") # must be used with "unique" mode

    args = parser.parse_args()
    main()

