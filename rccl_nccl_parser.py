import os
import sys
import argparse
import pandas as pd
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

coll_op_map = {
            "Broadcast": "broadcast_perf",
            "Reduce": "reduce_perf",
            "AllGather": "all_gather_perf",
            "ReduceScatter": "reduce_scatter_perf",
            "AllReduce": "all_reduce_perf",
            "Gather": "gather_perf",
            "Scatter": "scatter_perf",
            "AllToAll": "alltoall_perf",
#            "AllToAllv": "alltoallv_perf",
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
        commands.append((test_cmd, int(nnranks), line))

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
    fs = open(filename, 'w')
    fs.write("sep=|")
    fs.write("\n")
    keys = counts_map.keys()
    for key in keys:
        fs.write(key + "|" + str(counts_map[key]))
        fs.write("\n")
    fs.close()
    print ("INFO: Dumped out the count of each command in a file named: {}".format(filename))

def get_unique_commands(commands_and_nranks_and_lines):
    unique_values = []
    counts_map = {}
    nranks_map = {}
    unique_lines = []
    for c_and_nr_and_line in commands_and_nranks_and_lines:
        cmd = c_and_nr_and_line[0]
        nranks = c_and_nr_and_line[1]
        line = c_and_nr_and_line[2]
        if (cmd not in unique_values):
            counts_map[cmd] = 1
            nranks_map[cmd] = nranks
            unique_values.append(cmd)
            unique_lines.append(line)
        else:
            counts_map[cmd] = counts_map[cmd] + 1
    assert len(counts_map) == len(nranks_map)
    for cmd in counts_map.keys():
        assert counts_map[cmd] % nranks_map[cmd] == 0
        counts_map[cmd] = int(counts_map[cmd] / nranks_map[cmd])
    return unique_values, counts_map, unique_lines



def generate_topo_script(commands, topo_info, output_script):
    filename = output_script + ".sh"
    fs = open(filename, "w")
    for j in range(len(commands)):
        fs.write(commands[j])
        fs.write("\n : ' \n")
        for line in topo_info[j]:
            fs.write(line)
            fs.write("\n")
        fs.write(" ' \n")
    fs.close()
    print("INFO: Dumped out the commands in a script named: {}".format(filename))
    
# ts1-sjc2-27:11585:11585 [0] NCCL INFO Broadcast: opCount 0 sendbuff 0x7f988f200400 recvbuff 0x7f988f200400 count 440 datatype 0 op 0 root 0 comm 0x7f905c000dc0 [nranks=4] stream 0x55aa0a8d78f0
def coll_grouping(nccl_lines):
    opCount, coll, count, datatype, op_type, nnranks, comm = [], [], [], [], [], [], []
    for j in range(len(nccl_lines)):
        line = nccl_lines[j]
        split_list = line.split(" ")
        opCount.append(split_list[split_list.index("opCount") + 1])
        coll.append(split_list[split_list.index("INFO") + 1].replace(":", ""))
        count.append(split_list[split_list.index("count") + 1])
        datatype.append(split_list[split_list.index("datatype") + 1])
        op_type.append(split_list[split_list.index("op") + 1])
        comm.append(split_list[split_list.index("comm") + 1])
        nnranks.append(next(item for item in split_list if 'nranks' in item).split("=")[1].replace("]", ""))
    dict = {'Collective': coll, 'opCount': opCount, 'datatype': datatype, 'count':count, 'op_type':op_type, 
            'nnranks':nnranks, 'comm':comm}  
    df = pd.DataFrame(dict) 
    return df


def deviceInfo_connectionInfo(logfile):
    def process_string(line):
        split_list = line.split("[")
        return [split_list[0], split_list[1].split("]")[0]]
    
    fs = open(log_file, 'r')
    lines = fs.readlines()
    fs.close()
    commnunicator, rank, nranks, cudaDev, busId = [], [], [], [], []
    start_rank, start_busid, end_rank, end_busid, connection = [], [], [], [], []
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if ("Init COMPLETE" in line and "busId" in line):
            split_list = line.split(" ")
            commnunicator.append(split_list[split_list.index("comm") + 1])
            rank.append(split_list[split_list.index("rank") + 1])
            nranks.append(split_list[split_list.index("nranks") + 1])
            cudaDev.append(split_list[split_list.index("cudaDev") + 1])
            busId.append(split_list[split_list.index("busId") + 1])
        if ("via" in line):
            split_list = line.split(" ")
            sr, sb = process_string(split_list[split_list.index(":") + 1]) # first device
            er, eb = process_string(split_list[split_list.index("->") + 1]) # second device
            start_rank.append(sr)
            start_busid.append(sb)
            end_rank.append(er)
            end_busid.append(eb)
            connection.append(split_list[split_list.index("via") + 1]) 
    dict_device = {'comm': commnunicator, 'rank':rank, 'nranks':nranks, 'cudaDev':cudaDev, 'busId':busId}  
    dict_connections = {'start_rank': start_rank, 'start_busid': start_busid, 'end_rank': end_rank, 'end_busid': end_busid, 'connection': connection}      

    return pd.DataFrame(dict_device), pd.DataFrame(dict_connections)

class DisjointSet(object): # https://stackoverflow.com/questions/3067529/a-set-union-find-algorithm

    def __init__(self):
        self.leader = {} # maps a member to the group's leader
        self.group = {} # maps a group leader to the group (which is a set)

    def add(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])
                
def buildGraph(graphs, connectionList): # add an input of connection list
    G = nx.DiGraph()
    node_pool = []
    for graph in graphs:
        if graph[0][0] not in node_pool: 
            G.add_node(graph[0][0])
            node_pool.append(graph[0][0])
        for node in graph[1]:
            if node not in node_pool: 
                G.add_node(node)
                node_pool.append(node)
            label = connectionList[(connectionList['start_busid'] == graph[0][0]) & (connectionList['end_busid'] == node)]['connection'].unique()[0]
            G.add_edge(graph[0][0], node, label=label)

    to_remove = []
    for edge in G.edges():
        if (G.has_edge(edge[1], edge[0]) == False): to_remove.append([edge[0], edge[1]])
    for pair in to_remove: G.remove_edge(pair[0], pair[1])
    edges = G.to_undirected().edges()
    
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='#ffff00')
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color='b', arrows = True)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=nx.get_edge_attributes(G,'label'))
    
    plt.show()
    parents = {}
    ds = DisjointSet()
    for edge in edges: ds.add(edge[0], edge[1])
    
    outputs = []
    for k, v in ds.group.items():
        print(v)
        outputs.append(str(v))
    return outputs
        
    


def generate_topo(new_commands, raw_commands, log_file, output_name):
    deviceList, connectionList = deviceInfo_connectionInfo(log_file)
    all_info = pd.merge(coll_grouping(nccl_lines), deviceList, on="comm")
    topo_info = []
    for command in raw_commands:
        split_list = command.split(" ")
        opCount = split_list[split_list.index("opCount") + 1]
        coll = split_list[split_list.index("INFO") + 1].replace(":", "")
        count = split_list[split_list.index("count") + 1]
        datatype = split_list[split_list.index("datatype") + 1]
        nnranks = next(item for item in split_list if 'nranks' in item).split("=")[1].replace("]", "")
        op_type = split_list[split_list.index("op") + 1]
        #### Filter
        selected_info = all_info[(all_info['Collective'] == coll) & (all_info['opCount'] == opCount)
                                 & (all_info['datatype'] == datatype) & (all_info['count'] == count) 
                                 & (all_info['nnranks'] == nnranks) & (all_info['op_type'] == op_type)] 
#         print(selected_info)
        stage_2 = pd.merge(selected_info, connectionList, left_on=['rank','busId'], right_on=['start_rank','start_busid'],how='left')
        graphs = []
        for _ , subgroup in stage_2.groupby(['busId']):
            graphs.append([list(subgroup['busId'].unique()), list(subgroup['end_busid'].unique())])

        print("=="*20)
        print(command)
        outputs = buildGraph(graphs, connectionList)
        topo_info.append(outputs)
    generate_topo_script(new_commands, topo_info, output_name)

def main():
    log_file = os.path.abspath(args.nccl_debug_log)
    nccl_lines = get_useful_info(log_file)
    commands_and_nranks_and_opCount_and_line = parse_nccl_log(nccl_lines)
    if (args.unique):
        new_commands, counts_map, opCounts, raw_commands = get_unique_commands(commands_and_nranks_and_opCount_and_line)
        generate_topo(new_commands, raw_commands, log_file, args.output_script_name + "_unique_topo")
        generate_script(new_commands, args.output_script_name + "_unique")
        dump_counts_map(counts_map, args.output_script_name + "_counts")
    ########## TO-DO ##########
#     else:
#         commands = list(zip(*commands_and_nranks))[0]
#         generate_script(commands, args.output_script_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nccl-debug-log", type=str, required=True, help="Log from app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL")
    parser.add_argument("--output-script-name", type=str, required=False, default="net_nccl_rccl", help="Output command script")
    parser.add_argument("--unique", action="store_true", default=False, help="Get only the unique commands.")

    args = parser.parse_args()
    main()
