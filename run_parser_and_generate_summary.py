import os
import sys
import argparse

def main():
    if not args.rocm and not args.cuda:
        raise AssertionError("Please specify what system you are using (either --cuda or --rocm).")
    
    debug_log = os.path.abspath(args.nccl_debug_log)
    
    ## Generate a script to run nccl/rccl tests.
    if args.legacy_device_grouping:
        gen_cmd = gen_cmd = "python rccl_nccl_parser.py --nccl-debug-log " + debug_log + " --output-script-name net --unique --legacy-device-grouping"
    else:
        gen_cmd = "python rccl_nccl_parser.py --nccl-debug-log " + debug_log + " --output-script-name net --unique"
        
    gen_cmd = gen_cmd +  " --MP {} --DP {}".format(args.MP, args.DP)
    if args.perf_optim:
        gen_cmd = gen_cmd + " --perf-optim"

    if os.system(gen_cmd):
        print ("ERROR: Failed to parse the log.")
        sys.exit(1)
    if args.perf_optim:
        if args.cuda:
            raise AssertionError("The current tool does not support CUDA systems yet.")
        else:
            from define_search_space import overall_optimization, HIP_VISIBLE_DEVICES_assignment
            MP_first_MP_groups, MP_first_DP_groups, DP_first_MP_groups, DP_first_DP_groups = overall_optimization(rocm=True, numa_mode=False, MP=args.MP, DP=args.DP)
            HIP_VISIBLE_DEVICES_assignment(os.path.join(os.path.dirname(os.path.realpath(__file__)), "net_unique_MP.sh"), "MP_first_MP_command", MP_first_MP_groups)
            HIP_VISIBLE_DEVICES_assignment(os.path.join(os.path.dirname(os.path.realpath(__file__)), "net_unique_DP.sh"), "MP_first_DP_command", MP_first_DP_groups)
            HIP_VISIBLE_DEVICES_assignment(os.path.join(os.path.dirname(os.path.realpath(__file__)), "net_unique_MP.sh"), "DP_first_MP_command", DP_first_MP_groups)
            HIP_VISIBLE_DEVICES_assignment(os.path.join(os.path.dirname(os.path.realpath(__file__)), "net_unique_DP.sh"), "DP_first_DP_command", DP_first_DP_groups)
    
    
    ## change directory to rccl-tests/nccl-tests
    if args.rocm:
        rccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rccl-tests")
        os.system("cp -r *.sh " + rccl_tests_path) ######################### TODO
        os.chdir(rccl_tests_path)
        if os.path.exists(os.path.join(rccl_tests_path, 'build', 'all_reduce_perf')):
            print("rccl-tests is already built.")
        else:
            if os.system("make > /dev/null 2>&1"):
                print("ERROR: Failed to install rccl-tests.")
                sys.exit(1)
                
        run_script_cmd = []    
        
        if args.perf_optim:
            run_script_cmd.append(["HSA_FORCE_FINE_GRAIN_PCIE=1 sh MP_first_MP_command.sh | tee MP_first_MP_rccl_tests.txt",
                                   "MP_first_MP_rccl_tests.txt"])
            run_script_cmd.append(["HSA_FORCE_FINE_GRAIN_PCIE=1 sh MP_first_DP_command.sh | tee MP_first_DP_rccl_tests.txt",
                                   "MP_first_DP_rccl_tests.txt"])
            run_script_cmd.append(["HSA_FORCE_FINE_GRAIN_PCIE=1 sh DP_first_MP_command.sh | tee DP_first_MP_rccl_tests.txt",
                                   "DP_first_MP_rccl_tests.txt"])
            run_script_cmd.append(["HSA_FORCE_FINE_GRAIN_PCIE=1 sh DP_first_DP_command.sh | tee DP_first_DP_rccl_tests.txt",
                                   "DP_first_DP_rccl_tests.txt"])
        else:
            run_script_cmd.append(["HSA_FORCE_FINE_GRAIN_PCIE=1 sh net_unique_topo.sh | tee topo_rccl_tests.txt",
                                   "topo_rccl_tests.txt"])
        
        for cmd_and_output_name in run_script_cmd:
            if os.system(cmd_and_output_name[0]):
                print ("ERROR: Unable to run rccl-tests properly.")
                sys.exit(1)
            os.system("mv {} ../".format(cmd_and_output_name[1]))
        os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

        print (os.getcwd())
        from generate_summary import generate_summary_and_time
        MP_first_MP_time = generate_summary_and_time("MP_first_MP_rccl_tests.txt", "net_counts_MP.csv", "MP_first_MP")
        MP_first_DP_time = generate_summary_and_time("MP_first_DP_rccl_tests.txt", "net_counts_DP.csv", "MP_first_DP")
        DP_first_MP_time = generate_summary_and_time("DP_first_MP_rccl_tests.txt", "net_counts_MP.csv", "DP_first_MP")
        DP_first_DP_time = generate_summary_and_time("DP_first_DP_rccl_tests.txt", "net_counts_DP.csv", "DP_first_DP")
    
        if (MP_first_MP_time + MP_first_DP_time) < (DP_first_MP_time + DP_first_DP_time):
            print ("INFO: We recommend you to adopt MP-first optimization strategy.") 
            print("MP groups for HIP_VISIBLE_DEVICES: ", MP_first_MP_groups)
            print("DP groups for HIP_VISIBLE_DEVICES: ", MP_first_DP_groups)
            print("With the optimization strategy, the total expected time on RCCL ops is ", MP_first_MP_time + MP_first_DP_time)
        else:
            print ("INFO: We recommend you to adopt DP-first optimization strategy.") 
            print("MP groups for HIP_VISIBLE_DEVICES: ", DP_first_MP_groups)
            print("DP groups for HIP_VISIBLE_DEVICES: ", DP_first_DP_groups)
            print("With the optimization strategy, the total expected time on RCCL ops is ", DP_first_MP_time + DP_first_DP_time)
            
        
 

    if args.cuda:
        nccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nccl-tests")
        os.system("cp net_unique_topo.sh " + nccl_tests_path)
        os.chdir(nccl_tests_path)
        if os.path.exists(os.path.join(nccl_tests_path, 'build', 'all_reduce_perf')):
            print("nccl-tests is already built.")
        else:
            if os.system("make > /dev/null 2>&1"):
                print("ERROR: Failed to install rccl-tests.")
                sys.exit(1)
        
        os.system("cat net_unique_topo.sh")
        run_script_cmd = "HSA_FORCE_FINE_GRAIN_PCIE=1 sh net_unique_topo.sh | tee topo_rccl_tests.txt"
        if os.system(run_script_cmd):
            print ("ERROR: Unable to run nccl-tests properly.")
            sys.exit(1)
        os.system("mv topo_rccl_tests.txt ../")
        os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

        print (os.getcwd())
        summary_cmd = "python generate_summary.py --log-file topo_rccl_tests.txt --output-file-name net_summary --count-file net_counts.csv"
        os.system(summary_cmd)
        print ("INFO: Finished dumping all data.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nccl-debug-log", type=str, required=True, \
                            help="NCCL/RCCL log after running app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL")
    parser.add_argument("--MP", type=int, required=True, help="Model parallelism degree")
    parser.add_argument("--DP", type=int, required=True, help="Data parallelism degree")
    parser.add_argument("--rocm", action="store_true", default=False, help="Run the tests on ROCm using rccl-tests")
    parser.add_argument("--cuda", action="store_true", default=False, help="Run the tests on CUDA using nccl-tests")
    parser.add_argument("--perf-optim", action="store_true", default=False, help="Performance optimization mode")
    parser.add_argument("--legacy-device-grouping", action="store_true", default=False, help="NCCL/RCCL log after running app with NCCL or RCCL 2.8 or below")

    args = parser.parse_args()
    import time
    start = time.time()
    main()
    end = time.time()
    print("INFO: Total elapsed time = {} seconds".format(end - start))
    
# python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp2.txt --MP 2 --DP 4 --perf-optim --rocm
"""
/topo_aware_opt/device_ordering_exp/nccl-rccl-parser-test/nccl-rccl-parser
INFO: Dumped out the count of each command in a file named: MP_first_MP.csv
INFO: Dumped out the count of each command in a file named: MP_first_DP.csv
INFO: Dumped out the count of each command in a file named: DP_first_MP.csv
INFO: Dumped out the count of each command in a file named: DP_first_DP.csv
INFO: We recommend you to adopt MP-first optimization strategy.
MP groups for HIP_VISIBLE_DEVICES:  [[0, 3], [1, 2], [4, 5], [6, 7]]
DP groups for HIP_VISIBLE_DEVICES:  [[0, 1, 5, 7], [3, 2, 4, 6]]
"""
# python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp4_log_newPR.txt --MP 4 --DP 2 --perf-optim --rocm
"""
INFO: Dumped out the count of each command in a file named: MP_first_MP.csv
INFO: Dumped out the count of each command in a file named: MP_first_DP.csv
INFO: Dumped out the count of each command in a file named: DP_first_MP.csv
INFO: Dumped out the count of each command in a file named: DP_first_DP.csv
INFO: We recommend you to adopt MP-first optimization strategy.
MP groups for HIP_VISIBLE_DEVICES:  [[0, 1, 2, 3], [4, 5, 7, 6]]
DP groups for HIP_VISIBLE_DEVICES:  [[0, 4], [1, 5], [2, 6], [3, 7]]
INFO: Total elapsed time = 598.0428431034088 seconds
"""
# python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp4_log.txt --rocm --legacy-device-grouping
# python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp4_log_newPR.txt --rocm

# TODO: make a function which provides suggestions for DeepSpeed setup


# Then, we further investigate how to implemnt this device grouping in the application level.
# For example, DeepSpped GPT-2 pretraining:

#     Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
#     use 2 GPUs to parallelize the model. The present function will
#     create 4 model parallel groups and 2 data parallel groups as:
#         4 model parallel groups:
#             [g0, g1], [g2, g3], [g4, g5], [g6, g7]
#         2 data parallel groups:
#             [g0, g2, g4, g6], [g1, g3, g5, g7]
#     Note that for efficiency, the caller should make sure adjacent ranks
#     are on the same DGX box. For example if we are using 2 DGX-1 boxes
#     with a total of 16 GPUs, rank 0 to 7 belong to the first box and
#     ranks 8 to 15 belong to the second box.

