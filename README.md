# Topology-aware nccl-rccl-parser
This tool is used for dumping out the rccl-tests/nccl-test commands directly from an application to identify any potential bottlenecks of scaling while using RCCL/NCCL modules when running a distributed applications.

To get started please clone the following repository: 
git clone --recursive https://github.com/ROCmSoftwarePlatform/nccl-rccl-parser.git

To run the tests, we use the following repositories:

* On ROCm: https://github.com/ROCmSoftwarePlatform/rccl-tests
* On CUDA: https://github.com/NVIDIA/nccl-tests.git

# Pre-requisites:
* RCCL/NCCL installed. 
* Clone this repo with 
  ```
  git clone --recursive https://github.com/ROCmSoftwarePlatform/nccl-rccl-parser.git
  ```
* Run installation script by 
  ```
  sh install.sh
  ```
* Install rccl-tests or nccl-tests.
  ```
  cd rccl-tests; make
  
  or
  
  cd nccl-tests; make
  ```
# How to use the tool:

### Run application and collect RCCL/NCCL Log:**

Firstly, make sure you are running the experiments of a distributed setup of an application.
Make sure to run the application for at least 1 iteration using the below two environment variables into a log file named nccl_debug_log.txt

**On CUDA:**
```
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL <application> |& tee nccl_debug_log.txt
```
**On ROCm:** (needed for PCIe P2P but not needed for GPUs connected by XGMI, [ref](https://github.com/ROCmSoftwarePlatform/rccl/issues/92#issuecomment-540696989))
```
HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL <application> |& tee nccl_debug_log.txt
```


### Automated way:

To gather the performance results once you have the debug log with you. Run the below command. 

On CUDA devices, use --cuda argument.

On ROCm devices, use --rocm argument.

With NCCL or RCCL 2.8 or below, the argument "--legacy-device-grouping" is required for device grouping in applications. 

Note: If you don't mention the arguments the automated script only dumps out the output data from the parser. 

**On ROCm:**

```
python run_parser_and_generate_summary.py --nccl-debug-log nccl_debug_log.txt --rocm --legacy-device-grouping --MP 4 --DP 2
```

```
python run_parser_and_generate_summary.py --nccl-debug-log nccl_debug_new_log.txt --rocm --MP 4 --DP 2
```

We also support the performance optimization mode on ROCm. The tool first collects the RCCL operations in your application for model parallelism (MP) and data parallelism (DP). Then, it utilizes rccl-tests as a proxy to obtain performance stats for each RCCL ops. With the topology information collected by ROCm-SMI (e.g. "rocm-smi --showtopoweight" and "rsmi_topo_get_numa_node_number"), the tool iterates over possible optimal settings of device ordering for HIP_VISIBLE_DEVICES and NUMA control (under development) in the defined search space (define_searc_space.py). Lastly, the tool suggests users the most optimal setting to run their applications.

```
python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp2.txt --MP 2 --DP 4 --perf-optim --rocm
```
The example outputs from the perf-optim mode are shown as below:
```
/topo_aware_opt/device_ordering_exp/nccl-rccl-parser-test/nccl-rccl-parser
INFO: Dumped out the count of each command in a file named: MP_first_MP.csv
INFO: Dumped out the count of each command in a file named: MP_first_DP.csv
INFO: Dumped out the count of each command in a file named: DP_first_MP.csv
INFO: Dumped out the count of each command in a file named: DP_first_DP.csv
INFO: We recommend you to adopt MP-first optimization strategy.
MP groups for HIP_VISIBLE_DEVICES:  [[0, 3], [1, 2], [4, 5], [6, 7]]
DP groups for HIP_VISIBLE_DEVICES:  [[0, 1, 5, 7], [3, 2, 4, 6]]
If you are using DeepSpeed for your DL applications, please specify 'HIP_VISIBLE_DEVICES=0,3,1,2,5,4,7,6 <executable>'
INFO: Total elapsed time of running this tool = 598.0428431034088 seconds
```


**On CUDA:**

```
python run_parser_and_generate_summary.py --nccl-debug-log nccl_debug_log.txt --cuda --legacy-device-grouping
```

### To run the tool manually step by step:

**Use Parser to dump out the test commands:**

Once the log is being collected, use the parser to dump out all the rccl/nccl test commands or just the unique commands with their respective counts of the workload.
Note: To dump out the unique commands use the --unique argument. 
Note: To dump out the commands for the applications with NCCL or RCCL 2.8 or below use --legacy-device-grouping argument. 
Optional parameters: output-script-name, unique

Here is the usage of the script

```
python rccl_nccl_parser.py --nccl-debug-log nccl_debug_log.txt --output-script-name net
(or)
python rccl_nccl_parser.py --nccl-debug-log nccl_debug_log.txt --output-script-name net --unique
(or)
python rccl_nccl_parser.py --nccl-debug-log nccl_debug_log.txt --output-script-name net --unique --legacy-device-grouping"
```

The first command dumps out all the rccl/nccl tests in the order they get executed in the application. (net_rccl_nccl.sh file).
The second command dumps out a script file with unique commands and a csv file with commands and its counts of each command. 

**Run rccl-tests/nccl-tests:**

Once you dump out the scripts, make sure to copy the script in nccl-tests/rccl-tests folder and run the script and gather the 
Inside nccl-tests/rccl-tests repository:

```sh net_unique_topo.sh |& tee topo_rccl_tests.txt```

Once you run the above script, the performance data of each command is redirected to a text file. 

**Generate Summary:**

Now the final step is to use the above performance log and generate a summary in the form of CSV file for each of the command. The command gives the average values for each command like Time(us), algBw, busBw (out-of-place and in-place). For pytorch please consider out of place options. 

To generate the summary, navigate to the tool nccl-rccl-parser:

```
python generate_summary.py --log-file topo_rccl_tests.txt --output-file-name net_summary --count-file net_counts.csv
```
This dumps out a csv file with performance data for further analysis. 

**Supported Collectives:**

Currently only the AllReduce and Broadcast calls are being supported by this tool. Based on running more experiments other collectives need to be added. 

