import os
import sys
import argparse

def main():
    debug_log = os.path.abspath(args.nccl_debug_log)
    
    ## Generate a script to run nccl/rccl tests.
    if args.legacy_device_grouping:
        gen_cmd = gen_cmd = "python rccl_nccl_parser.py --nccl-debug-log " + debug_log + " --output-script-name net --unique --legacy-device-grouping"
    else:
        gen_cmd = "python rccl_nccl_parser.py --nccl-debug-log " + debug_log + " --output-script-name net --unique"
    if os.system(gen_cmd):
        print ("ERROR: Failed to parse the log.")
        sys.exit(1)
      
    ## change directory to rccl-tests/nccl-tests
    if args.rocm:
        rccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rccl-tests")
        os.system("cp net_unique_topo.sh " + rccl_tests_path)
        os.chdir(rccl_tests_path)
        if os.path.exists(os.path.join(os.getcwd(), 'build', 'all_reduce_perf')):
            print("rccl-tests is already built.")
        else:
            if os.system("make > /dev/null 2>&1"):
                print("ERROR: Failed to install rccl-tests.")
                sys.exit(1)
        #### TODO: if rccl-tests is already built ####
        os.system("cat net_unique_topo.sh")
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
        if os.path.exists(os.path.join(os.getcwd(), 'build', 'all_reduce_perf')):
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
    parser.add_argument("--rocm", action="store_true", default=False, help="Run the tests on ROCm using rccl-tests")
    parser.add_argument("--cuda", action="store_true", default=False, help="Run the tests on CUDA using nccl-tests")
    parser.add_argument("--legacy-device-grouping", action="store_true", default=False, help="NCCL/RCCL log after running app with NCCL or RCCL 2.8 or below")

    args = parser.parse_args()
    main()
# python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp4_log.txt --rocm --legacy-device-grouping
# python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp4_log_newPR.txt --rocm

# TODO: Check if nccl-tests/rccl-tests is already built or not. (OK)
# TODO: Iterate different sets of device grouping.
# TODO: suggest users the most optimal setting for device grouping


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

