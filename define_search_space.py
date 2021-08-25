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

def main():
    deviceIdMapping_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "deviceIdMapping")
    if os.path.exists(os.path.join(os.path.realpath(__file__), 'deviceIdMapping', 'busId_HIP_map.txt')):
        os.chdir(deviceIdMapping_path)
        os.system("make")
    
    busId_HIP_map = os.path.join(deviceIdMapping_path, "busId_HIP_map.txt")
    fs = open(busId_HIP_map, 'r')
    lines = fs.readlines()
    fs.close()
    for line in lines:
        
        
    if args.rocm:
        rccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rccl-tests")
        os.system("cp net_unique_topo.sh " + rccl_tests_path)
        os.chdir(rccl_tests_path)
    if args.cuda:
        nccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nccl-tests")
        os.system("cp net_unique_topo.sh " + nccl_tests_path)
        os.chdir(nccl_tests_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocm", action="store_true", default=False, help="Run the tests on ROCm using rccl-tests")
    parser.add_argument("--cuda", action="store_true", default=False, help="Run the tests on CUDA using nccl-tests")
    parser.add_argument("--numa", action="store_true", default=False, help="Expand the search space by adding NUMA control.")
    parser.add_argument("--TODO", action="store_true", default=False, help="TODO")
    args = parser.parse_args()
    main()
# python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp4_log.txt --rocm --legacy-device-grouping
# python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp4_log_newPR.txt --rocm

