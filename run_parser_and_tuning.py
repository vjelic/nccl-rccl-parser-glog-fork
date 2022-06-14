import os
import sys
import argparse

def main():
    debug_log = os.path.abspath(args.nccl_debug_log)

    ##### Firstly call rccl_nccl_parser.py to parse the ......log.txt file
    ## Generate a script to run nccl/rccl tests.
    gen_cmd = "python rccl_nccl_parser.py --nccl-debug-log " + debug_log + " --output-script-name net --unique"
    if os.system(gen_cmd):
        print ("ERROR: Failed to parse the log.")
        sys.exit(1)

    ## change directory to rccl-tests/nccl-tests
    if args.rocm:
        rccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rccl-tests")
        os.system("cp net_counts_new.csv " + rccl_tests_path)
        os.chdir(rccl_tests_path)
        if os.system("./install.sh > /dev/null 2>&1"):
            print("ERROR: Failed to install rccl-tests.")
            sys.exit(1)

        # os.system("cat net_unique.sh")
        # python brute_force_search_workload.py --gpu-per-node 8 --test-iteration 100 --net_counts_path net_counts.csv --output-csv-name mlperf_resnet50_2H4P.csv
        # TODO: Check if there is any op which has nranks != gpu_per_node
        # TODO: Consolidate similar rccl ops to save brute-force search time
        run_script_cmd = "python ../brute_force_search_workload.py --gpus-per-node 8 --test-iteration 100 --net-counts-path net_counts_new.csv --output-csv-name mlperf_resnet50_2H4P.csv"
        if os.system(run_script_cmd):
            print ("ERROR: Unable to run rccl-tests properly.")
            sys.exit(1)
        os.system("mv mlperf_resnet50_2H4P.csv ../")
        os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

        print (os.getcwd())
        # summary_cmd = "python generate_summary.py --log-file rccl_perf_log.txt --script-file net_unique.sh --count-file net_counts.csv"
        # os.system(summary_cmd)
        print ("INFO: Finished dumping all data.")
    else:
        print ("ERROR: The current tool only supports on ROCm.")
        sys.exit(1)

    # if args.cuda:
    #     nccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nccl-tests")
    #     os.system("cp net_unique.sh " + nccl_tests_path)
    #     os.chdir(nccl_tests_path)
    #     if os.system("make > /dev/null 2>&1"):
    #         print ("ERROR: Failed to install nccl-unit tests")
    #         sys.exit(1)

    #     os.system("cat net_unique.sh")
    #     run_script_cmd = "sh net_unique.sh | tee nccl_perf_log.txt"
    #     if os.system(run_script_cmd):
    #         print ("ERROR: unable to run nccl-tests")
    #         sys.exit(1)
    #     os.system("mv nccl_perf_log.txt ../")
    #     os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

    #     summary_cmd = "python generate_summary.py --log-file nccl_perf_log.txt --script-file net_unique.sh --output-file-name nv_net_summary --count-file net_counts.csv"
    #     os.system(summary_cmd)
    #     print ("INFO: Finished dumping all data.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nccl-debug-log", type=str, required=True, \
                            help="NCCL/RCCL log after running app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL")
    parser.add_argument("--rocm", action="store_true", default=False, help="Run the tests on ROCm using rccl-tests")
    # parser.add_argument("--cuda", action="store_true", default=False, help="Run the tests on CUDA using nccl-tests")

    args = parser.parse_args()
    main()

# python run_parser_and_tuning.py --nccl-debug-log RN50_AMP_LARS_8GPUS_MI200_loss_log.txt --rocm
# python brute_force_search_workload.py --gpu-per-node 8 --test-iteration 100 --net_counts_path net_counts.csv --output-csv-name mlperf_resnet50_2H4P.csv
