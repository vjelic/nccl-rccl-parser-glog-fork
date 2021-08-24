/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <iomanip>
#include "hip/hip_runtime.h"
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <stdint.h>
#include <assert.h>
#include "rocm_smi/rocm_smi.h"
// #include "rocm_smi/rocm_smi_exception.h"


#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define failed(...)                                                                                \
    printf("%serror: ", KRED);                                                                     \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("error: TEST FAILED\n%s", KNRM);                                                        \
    exit(EXIT_FAILURE);

#define HIPCHECK(error)                                                                            \
    if (error != hipSuccess) {                                                                     \
        printf("%serror: '%s'(%d) at %s:%d%s\n", KRED, hipGetErrorString(error), error, __FILE__,  \
               __LINE__, KNRM);                                                                    \
        failed("API returned error code.");                                                        \
    }

std::string int2hex(uint64_t num, int digits) {
    std::stringstream sstream;
    sstream << std::setfill('0') << std::setw(digits) << std::hex << num;
    std::string result = sstream.str();
    return result;
}


int main(int argc, char* argv[]) {
    int deviceCnt;
    HIPCHECK(hipGetDeviceCount(&deviceCnt));
    std::cout << "====================== Number of HIP visible devices: " << deviceCnt << std::endl;
    /* Example outputs from rocm-smi --showbus

    GPU[0]              : PCI Bus: 0000:43:00.0
    GPU[1]              : PCI Bus: 0000:23:00.0
    GPU[2]              : PCI Bus: 0000:26:00.0
    GPU[3]              : PCI Bus: 0000:03:00.0
    GPU[4]              : PCI Bus: 0000:E3:00.0
    GPU[5]              : PCI Bus: 0000:C3:00.0
    GPU[6]              : PCI Bus: 0000:C6:00.0
    GPU[7]              : PCI Bus: 0000:83:00.0
    */

    // ROCm-smi
    rsmi_status_t ret;
    uint32_t num_devices;
    ret = rsmi_init(0);
    ret = rsmi_num_monitor_devices(&num_devices);
    std::cout << "====================== Number of GPUs on your machine which can be observed by ROCm-SMI: "<< num_devices << std::endl;
    std::cout << "====== ROCm-SMI device ID ======= PCI bus ID ======= HIP device ID ========================================================== NUMA node ====" << std::endl;
        
    for (int i = 0; i < num_devices; i++) {
        uint64_t val_ui64; // bdfid in rocm_smi.cc
        rsmi_status_t  err = rsmi_dev_pci_id_get(i, &val_ui64);
        if (err != RSMI_STATUS_SUCCESS) {
            std::cout << "Failed to get PCI ID from ROCm-SMI." << std::endl;
            return err;
        }
        uint32_t numa_node;
        rsmi_status_t  err1 = rsmi_topo_get_numa_node_number(i, &numa_node);
        if (err1 != RSMI_STATUS_SUCCESS) {
            std::cout << "Failed to get NUMA node number from ROCm-SMI." << std::endl;
            return err;
        }
        auto domain = (val_ui64 >> 32) & 0xffff;
        auto bus = (val_ui64 >> 8) & 0xff;
        auto device = (val_ui64 >> 3) & 0x1f;
        auto function = val_ui64 & 0x7;
        std::string pciString = int2hex(domain, 4) + ":" + int2hex(bus, 2) + ":" + int2hex(device, 2) + "." + int2hex(function, 1);
        const char* busIdStr = (pciString).c_str();
        int hipDeviceId;
        std::cout << "                " << i << "                "<< pciString << "              ";
        if (hipDeviceGetByPCIBusId(&hipDeviceId, busIdStr) != hipSuccess) {
            std::cout << "N/A (cannot map PCI Bus ID: " << busIdStr << " to a HIP visible device)      "<< numa_node << std::endl;
        } else std::cout << hipDeviceId << "                                                                      " << numa_node <<std::endl;

    }
    ret = rsmi_shut_down();
}

/*
 We also need to use hip_rocm_smi_mapping.cpp (with HIP_API and ROCm-SMI lib) to get HIP device ids, topology information (including weight between two GPUs and NUMA node for each GPU). 
 $rocm-smi --showtopoweight ()
 $rocm-smi --showtoponuma ()
 
 https://github.com/RadeonOpenCompute/rocm_smi_lib/blob/master/src/rocm_smi.cc#L3385
 TODO: this script only works for ROCm systems.********************
 
 rsmi_topo_get_link_weight(uint32_t dv_ind_src, uint32_t dv_ind_dst,
                          uint64_t *weight) {
  TRY

  uint32_t dv_ind = dv_ind_src;
  GET_DEV_AND_KFDNODE_FROM_INDX
  DEVICE_MUTEX

  if (weight == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

*/
