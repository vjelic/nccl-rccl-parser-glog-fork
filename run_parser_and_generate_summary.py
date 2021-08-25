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
        ## TODO: Run define_search_space.py
        #        expect to get several sets of HIP_VISIBLE_DEVICES device groups for MP and DP
        #                                    e.g. MP = 2, DP = 4  
        #                                         #0 -> [[0,1], [2,3], [4,5], [6,7]], [[0,2,4,6],[1,3,5,7]]
        #                                         #1 -> [[0,2], [1,3], [4,6], [5,7]], [[0,1,4,5],[2,3,6,7]]                                        
        #                                         #?
        gen_cmd = gen_cmd + " --perf-optim"
    if os.system(gen_cmd):
        print ("ERROR: Failed to parse the log.")
        sys.exit(1)
    
    ## change directory to rccl-tests/nccl-tests
    if args.rocm:
        rccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rccl-tests")
        os.system("cp net_unique_topo.sh " + rccl_tests_path)
        os.chdir(rccl_tests_path)
        if os.path.exists(os.path.join(rccl_tests_path, 'build', 'all_reduce_perf')):
            print("rccl-tests is already built.")
        else:
            if os.system("make > /dev/null 2>&1"):
                print("ERROR: Failed to install rccl-tests.")
                sys.exit(1)
                
        if args.perf_optim:
            # TODO
            # net_unique_MP.sh
            # net_unique_DP.sh
            run_script_cmd = "HSA_FORCE_FINE_GRAIN_PCIE=1 sh net_unique_topo.sh | tee topo_rccl_tests.txt"
        else:
            run_script_cmd = "HSA_FORCE_FINE_GRAIN_PCIE=1 sh net_unique_topo.sh | tee topo_rccl_tests.txt"

        if os.system(run_script_cmd):
            print ("ERROR: Unable to run rccl-tests properly.")
            sys.exit(1)
        os.system("mv topo_rccl_tests.txt ../")
        os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

        print (os.getcwd())
        summary_cmd = "python generate_summary.py --log-file topo_rccl_tests.txt --output-file-name net_summary --count-file net_counts.csv"
                
        os.system(summary_cmd)
        print ("INFO: Finished dumping all data.")
        
        

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
    main()
# python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp4_log.txt --rocm --legacy-device-grouping
# python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp4_log_newPR.txt --rocm

# TODO: Check if nccl-tests/rccl-tests is already built or not. (OK)
# TODO: Add the perf_optim mode
# TODO: Iterate different sets of device grouping.
# TODO: Suggest users the most optimal setting for device grouping


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

