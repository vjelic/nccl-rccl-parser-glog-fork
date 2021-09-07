"""
The purpose of this script is to define search space for topology-aware optimization
1. HIP_VISIBLE_DEVICES (default search space)
2. NUMA binding
3. rccl env var?
4. ...


We first read "deviceIdMapping/busId_HIP_map.txt"

# TODO: Read deviceIdMapping/busId_HIP_map.txt 
# TODO: default: search space for CUDA_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES
# TODO: Iterate different sets of device grouping.
# TODO: suggest users the most optimal setting for device grouping
"""

import os
import sys
import argparse
import numpy as np ## TODO: shall we check if this is install in install.sh?
import itertools as it
if os.path.exists("/opt/rocm/bin/rocm-smi") == False:
        raise AssertionError("Please make sure ROCm-SMI is installed.")
sys.path.append("/opt/rocm/bin")
import rocm_smi
from rsmiBindings import *

def rocmsmi_hip_id_mapping(path_to_deviceIdMapping):
    fs = open(path_to_deviceIdMapping, 'r')
    lines = fs.readlines()
    fs.close()
    rocmsmi_hip_id_map = {}
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if "=" not in line:
            split_list = line.split()
            rocmsmi_hip_id_map[ord(split_list[0]) - ord('0')] = ord(split_list[2]) - ord('0')
    return rocmsmi_hip_id_map

# rocm-smi --showtopoweight
# Older version of ROCm might not have rsmi_topo_get_link_weight function...
def showWeightTopology(deviceList, link_weights, rocmsmi_hip_id_map): 
    for srcdevice in deviceList:
        for destdevice in deviceList:
            if (srcdevice == destdevice):
                link_weights[rocmsmi_hip_id_map[srcdevice]][rocmsmi_hip_id_map[destdevice]] = 0
                continue
            weight = c_uint64()
            ret = rocmsmi.rsmi_topo_get_link_weight(srcdevice, destdevice, byref(weight))
            if rocm_smi.rsmi_ret_ok(ret):
                link_weights[rocmsmi_hip_id_map[srcdevice]][rocmsmi_hip_id_map[destdevice]] = weight.value
            else:
                rocm_smi.printErrLog(srcdevice, 'Cannot read Link Weight: Not supported on this machine', None)
    return link_weights

# rocm-smi --showtoponuma
# Older version of ROCm might not have rsmi_topo_get_link_weight function...
def showNumaTopology(deviceList, numa_nodes, rocmsmi_hip_id_map): 
    """ Display the HW Topology Information for numa nodes
    This reads the HW Topology file and display the matrix for the nodes
    @param deviceList: List of DRM devices (can be a single-item list)
    """
    numa_numbers = c_uint32()
    for device in deviceList:
        ret = rocmsmi.rsmi_topo_get_numa_node_number(device, byref(numa_numbers))
        if rocm_smi.rsmi_ret_ok(ret, device):
            numa_nodes[rocmsmi_hip_id_map[device]] = numa_numbers.value
        else:
            rocm_smi.printErrLog(device, "Cannot read Numa Node")
    return numa_nodes

def collect_topo(path_to_deviceIdMapping):
    # rocmsmi_hip_id_map[rocmsmi_id] = hip_id
    rocmsmi_hip_id_map = rocmsmi_hip_id_mapping(path_to_deviceIdMapping)
    deviceList = rocm_smi.listDevices()
    devices_ind = range(len(deviceList))
    link_weights = np.zeros(shape=(len(devices_ind),len(devices_ind)), dtype=np.uint32)
    numa_nodes = np.zeros(shape=(len(devices_ind)), dtype=np.uint32)
    return showWeightTopology(deviceList, link_weights, rocmsmi_hip_id_map), showNumaTopology(deviceList, numa_nodes, rocmsmi_hip_id_map)


############## How if (N % MP) != 0? ############## TODO
############## How if (N % MP) != 0? ############## TODO
############## How if (N % MP) != 0? ############## TODO: Add one more layer? N = 4  but we have 8 GPUs 
"""
(0, 1, 2, 3)
(0, 1, 2, 4)
(0, 1, 2, 5)
(0, 1, 2, 6)
.
.
.
C8_4?
"""
# Ref: https://stackoverflow.com/a/45645704

# If it is an iterable, it must be an iterable of integers in the range 0 <= x < 256, which are used as the initial contents of the array.


"""
The first priority of optimization strategy.
If MP = 2, DP = 4 and MP operations are more "costly"(metric?) in a DL application, we need to  
optimize_device_group_step1(MP, DP, topo_matrix)
optimize_device_group_step2(MP, DP, topo_matrix)
*outputs: minSum_group_MP_final, minCom_group_MP_final, minSum_group_DP_final, minCom_group_DP_final

However, if DP operations are more costly in a DL application, we need to swap the arguments MP and DP.
optimize_device_group_step1(DP, MP, topo_matrix)
optimize_device_group_step2(DP, MP, topo_matrix)
*outputs: minSum_group_DP_final, minCom_group_DP_final, minSum_group_MP_final, minCom_group_MP_final
"""



def findBestGroup_step1(result, topo_matrix, N, degree_1):
    def findBestSubgroup(subgroup, topo_matrix, degree_1):
        candidates = list(it.permutations(subgroup, degree_1)) ###
        minSum = float('inf')
        minCom = ""
        for candidate in candidates:
            tempSum = 0
            for i in range(len(candidate) - 1):
                tempSum += topo_matrix[ord(candidate[i]) - ord('a')][ord(candidate[i + 1]) - ord('a')]
            tempSum += topo_matrix[ord(candidate[len(candidate) - 1]) - ord('a')][ord(candidate[0]) - ord('a')] ## for allreduce, allgather
            if tempSum < minSum:
                minSum = tempSum
                # minCom = procesResults(candidate)
                minCom = candidate
        return minSum, minCom
    
    minSum_group = float('inf')
    minCom_group = []
    for group in result: # ('ab', 'cd', 'ef', 'gh')
        tempSum = 0
        tempCom = []
        for subgroup in group: # 'ab'
            minSum, minCom = findBestSubgroup(subgroup, topo_matrix, degree_1)
            tempSum += minSum
            tempCom.append(minCom)
        if tempSum <= minSum_group:  ## Shall we collect the possibility with the same minimum sum of link weights?
            if tempSum == minSum_group:
                minCom_group.append([tempSum, tempCom])
            else:
                minSum_group = tempSum
                minCom_group = [[tempSum, tempCom]]
    return minCom_group

def optimize_device_group_step1(degree_1, degree_2, topo_matrix): 
    def unique_group(iterable, number_of_groups, group_size):
        """Return an iterator, comprising groups of size `k` with combinations of size `n`."""
        # Build separate combinations of `group_size` characters
        groups = ("".join(i) for i in it.combinations(iterable, group_size))    # 'AB', 'AC', 'AD', ...
        # Build unique groups of `number_of_groups` by keeping the longest sets of characters
        temp = (i for i in it.combinations(groups, number_of_groups) 
                if len(set("".join(i))) == sum((map(len, i))))     # ('AB', 'CD'), ('AB', 'CE'), ... 
        return list(temp)
    N = degree_1 * degree_2
    assert N % degree_1 == 0
    start = ord('a')
    iterable = ""
    for i in range(N):
        iterable += chr(start + i) 
    result = unique_group(iterable, degree_2 , degree_1) # 4
    minCom_group = findBestGroup_step1(result, topo_matrix, N, degree_1)
    return minCom_group


def findBestGroup_step2(step2_candidates, topo_matrix, degree_1, degree_2, used_device, tempSum, tempString, minSum, minString, numGroups, minCom_group):
    if len(used_device) == degree_1*degree_2 and numGroups == degree_1:
        if tempSum <= minSum:  ## Shall we collect the possibility with the same minimum sum of link weights?
            if tempSum == minSum:
                minCom_group.append([tempSum, tempString])
            else:
                minSum = tempSum
                minCom_group = [[tempSum, tempString]]
            return minSum                
    for group in step2_candidates:
        used = False
        weights_sum = 0
        for i in group:
            if i in used_device: # we cannot choose this group
                used = True
                break
        if used == False:
            for i in group:
                used_device.append(i)
            for i in range(len(group) - 1):
                weights_sum += topo_matrix[ord(group[i]) - ord('a')][ord(group[i + 1]) - ord('a')]
                tempString += str(ord(group[i]) - ord('a'))
            weights_sum += topo_matrix[ord(group[len(group) - 1]) - ord('a')][ord(group[0]) - ord('a')] ## for allreduce, allgather
            tempString += str(ord(group[len(group) - 1]) - ord('a'))
            
            tempSum += weights_sum
            minSum = findBestGroup_step2(step2_candidates, topo_matrix, degree_1, degree_2, used_device, tempSum, tempString, minSum, minString, numGroups + 1, minCom_group)
            tempSum -= weights_sum
            tempString = tempString[:-len(group)]
            
            for i in group:
                # used_device.remove(i)
                used_device.pop()
    return minSum

def procesResults(minCom_group_step1, minCom_group_step2, degree_2):
    results_step1, results_step2 = [], []
    for i in range(len(minCom_group_step1)):
        subgroup = []
        for j in range(len(minCom_group_step1[i])):
            subgroup.append(ord(minCom_group_step1[i][j]) - ord('a'))
        results_step1.append(subgroup)
    
    subgroup = []
    for i in range(len(minCom_group_step2)):
        subgroup.append(ord(minCom_group_step2[i]) - ord('0'))
        if i % degree_2 == degree_2 - 1:
            results_step2.append(subgroup)            
            subgroup = []
    return results_step1, results_step2

def optimize_device_group_step2(degree_1, degree_2, topo_matrix):
    minCom_group_step1 = optimize_device_group_step1(degree_1, degree_2, topo_matrix)
    minCom_group_step1_final = minCom_group_step2_final = [] 
    minSum_group_step1_final = minSum_group_step2_final = float('inf')
    
    for step1_group in minCom_group_step1:
        step2_candidates = []
        for k in it.product(*step1_group[1]):
            step2_candidates.append(k)
        minCom_group = []
        # results = findBestGroup_step2(step2_candidates, topo_matrix, degree_1, DP, used_device, 0, "", minSum, minString, numGroups)
        results = findBestGroup_step2(step2_candidates, topo_matrix, degree_1, degree_2, [], 0, "", float('inf'), "", 0, minCom_group)
        if minSum_group_step2_final > minCom_group[0][0]:
            minSum_group_step1_final, minCom_group_step1_final = step1_group[0], step1_group[1]
            minSum_group_step2_final, minCom_group_step2_final = minCom_group[0][0], minCom_group[0][1]
    minCom_group_step1_final, minCom_group_step2_final = procesResults(minCom_group_step1_final, minCom_group_step2_final, degree_2)
    return minSum_group_step1_final, minCom_group_step1_final, minSum_group_step2_final, minCom_group_step2_final

# The default mode is to utilize HIP_VISIBLE_DEVICES ordering to optimize performance.
# TODO: add NUMA mode.

def overall_optimization(rocm, numa_mode, MP, DP):
    rocm_smi.initializeRsmi()
    deviceIdMapping_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "deviceIdMapping")
    if os.path.exists(os.path.join(os.path.realpath(__file__), 'deviceIdMapping', 'busId_HIP_map.txt')):
        os.chdir(deviceIdMapping_path)
        os.system("make")
    
    path_to_deviceIdMapping = os.path.join(deviceIdMapping_path, "busId_HIP_map.txt")
    if MP == 0 or DP == 0:
        # we do not need to consider HIP_VISIBLE_DEVICES ordering
        if MP != 0:
            assert MP <= len(rocm_smi.listDevices())
        else:
            assert DP <= len(rocm_smi.listDevices())
        
    if MP != 0 and DP != 0:
        # Make sure the number of GPUs in a machine is greater than or equal to MP * DP
        assert MP * DP <= len(rocm_smi.listDevices())
        link_weights, numa_nodes = collect_topo(path_to_deviceIdMapping)
        MP_first_MP_sum, MP_first_MP_groups, MP_first_DP_sum, MP_first_DP_groups = optimize_device_group_step2(
            degree_1 = MP, degree_2 = DP, topo_matrix = link_weights)
        print("========="*4)
        print("MP-first optimization strategy")
        print(MP_first_MP_sum, MP_first_MP_groups, " ==> Step 1 (MP groups)")
        print(MP_first_DP_sum, MP_first_DP_groups, " ==> Step 2 (DP groups)")
        
        print("========="*4)
        print("DP-first optimization strategy")
        DP_first_DP_sum, DP_first_DP_groups, DP_first_MP_sum, DP_first_MP_groups = optimize_device_group_step2(
            degree_1 = DP, degree_2 = MP, topo_matrix = link_weights)
        print(DP_first_MP_sum, DP_first_MP_groups, " ==> Step 2 (MP groups)")
        print(DP_first_DP_sum, DP_first_DP_groups, " ==> Step 1 (DP groups)")
        
    return MP_first_MP_groups, MP_first_DP_groups, DP_first_MP_groups, DP_first_DP_groups
        
        ## Use these HIP_VISIBLE_DEVICES ids for net_unique_MP.sh and net_unique_DP.sh
        ## TODO: net_counts.csv split into net_counts_MP.csv and net_counts_DP.csv
        
        # if args.numa:
        
def HIP_VISIBLE_DEVICES_assignment(path_to_command_script, output_name, groups):
    fs = open(path_to_command_script, 'r')
    lines = fs.readlines()
    fs.close()
    filename = output_name + ".sh"
    with open(filename, 'w') as f:
        for j in range(1, len(lines)): # skip first line
            f.write("echo '==========================================================' \n")
            for group in groups: # not work for legacy_device_grouping, "device_group_list" is list of sets
                device_setting = "HIP_VISIBLE_DEVICES="
                for k, device in enumerate(group):
                    if k == len(group) - 1:
                        device_setting = device_setting + str(device) + " "
                    else:                   
                        device_setting = device_setting + str(device) + ","
                device_setting = device_setting + " " + lines[j]                    
                f.write(device_setting) # TODO: generate_summary.py also need to be refactored. (def parse_nccl_performance(perf_lines,...)
        f.write("echo '==========================================================' \n") ## TODO

    

def main():
    deviceIdMapping_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "deviceIdMapping")
    if os.path.exists(os.path.join(os.path.realpath(__file__), 'deviceIdMapping', 'busId_HIP_map.txt')):
        os.chdir(deviceIdMapping_path)
        os.system("make")
    
    path_to_deviceIdMapping = os.path.join(deviceIdMapping_path, "busId_HIP_map.txt")
    if args.MP == 0 or args.DP == 0:
        # we do not need to consider HIP_VISIBLE_DEVICES ordering
        if args.MP != 0:
            assert args.MP <= len(rocm_smi.listDevices())
        else:
            assert args.DP <= len(rocm_smi.listDevices())
        print("TODO")
        
    if args.MP != 0 and args.DP != 0:
        # Make sure the number of GPUs in a machine is greater than or equal to MP * DP
        assert args.MP * args.DP <= len(rocm_smi.listDevices())
        link_weights, numa_nodes = collect_topo(path_to_deviceIdMapping)
        MP_first_MP_sum, MP_first_MP_groups, MP_first_DP_sum, MP_first_DP_groups = optimize_device_group_step2(
            degree_1 = args.MP, degree_2 = args.DP, topo_matrix = link_weights)
        print("========="*4)
        print("MP-first optimization strategy")
        print(MP_first_MP_groups, MP_first_MP_sum, " ==> Step 1 (MP groups)")
        print(MP_first_DP_groups, MP_first_DP_sum, " ==> Step 2 (DP groups)")
        
        print("========="*4)
        print("DP-first optimization strategy")
        DP_first_DP_sum, DP_first_DP_groups, DP_first_MP_sum, DP_first_MP_groups 
        minSum_group_step1_final, minCom_group_step1_final, minSum_group_step2_final, minCom_group_step2_final = optimize_device_group_step2(
            degree_1 = args.DP, degree_2 = args.MP, topo_matrix = link_weights)
        print(DP_first_MP_sum, DP_first_MP_groups, " ==> Step 2 (MP groups)")
        print(DP_first_DP_sum, DP_first_DP_groups, " ==> Step 1 (DP groups)")
        
        return MP_first_MP_groups, MP_first_DP_groups, DP_first_MP_groups, DP_first_DP_groups
        ## Use these HIP_VISIBLE_DEVICES ids for net_unique_MP.sh and net_unique_DP.sh
        ## TODO: net_counts.csv split into net_counts_MP.csv and net_counts_DP.csv
        
        
        # if args.numa:
        
        
#     fs = open(busId_HIP_map, 'r')
#     lines = fs.readlines()
#     fs.close()
#     for line in lines:
        
        
#     if args.rocm:
#         rccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rccl-tests")
#         os.system("cp net_unique_topo.sh " + rccl_tests_path)
#         os.chdir(rccl_tests_path)
#     if args.cuda:
#         nccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nccl-tests")
#         os.system("cp net_unique_topo.sh " + nccl_tests_path)
#         os.chdir(nccl_tests_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocm", action="store_true", default=False, help="Run the tests on ROCm using rccl-tests")
    parser.add_argument("--cuda", action="store_true", default=False, help="Run the tests on CUDA using nccl-tests")
    parser.add_argument("--numa", action="store_true", default=False, help="Expand the search space by adding NUMA control.")
    parser.add_argument("--MP", type=int, required=True, help="Model parallelism degree")
    parser.add_argument("--DP", type=int, required=True, help="Data parallelism degree")
    args = parser.parse_args()
    if os.path.exists("/opt/rocm/bin/rocm-smi") == False:
        raise AssertionError("Please make sure ROCm-SMI is installed.")
    sys.path.append("/opt/rocm/bin")
    import rocm_smi
    from rsmiBindings import *
    rocm_smi.initializeRsmi()
    main()
    
    MP_first_MP_groups, MP_first_DP_groups, DP_first_MP_groups, DP_first_DP_groups = overall_optimization(rocm=True, numa_mode=False, MP=2, DP=4)

    path_to_command_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), "net_unique_MP.sh")
    output_name = "MP_first_MP_command"
    HIP_VISIBLE_DEVICES_assignment(path_to_command_script, output_name, MP_first_MP_groups)
    HIP_VISIBLE_DEVICES_assignment(os.path.join(os.path.dirname(os.path.realpath(__file__)), "net_unique_DP.sh"), "MP_first_DP_command", MP_first_DP_groups)
# python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp4_log.txt --rocm --legacy-device-grouping
# python define_search_space.py --MP 2 --DP 4
