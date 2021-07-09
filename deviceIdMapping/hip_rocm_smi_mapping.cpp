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
#include <regex>
#include <vector>

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

void printCompilerInfo() {
#ifdef __HCC__
    printf("compiler: hcc version=%s, workweek (YYWWD) = %u\n", __hcc_version__, __hcc_workweek__);
#endif
#ifdef __NVCC__
    printf("compiler: nvcc\n");
#endif
}


int64_t busIdToInt64(std::string busId) {
    int64_t id;
    const int size = busId.size();
    std::string hexStr;
    int hexOffset = 0;
    for (int i= 0; i < size; i++) {
        char c = busId[i];
        if (c == '.' || c == ':') continue;
        if ((c >= '0' && c <= '9') ||
            (c >= 'A' && c <= 'F') ||
            (c >= 'a' && c <= 'f')) {
                hexStr += busId[i];
        } else break;
    }
    std::stringstream ss;
    ss << std::hex << hexStr;
    ss >> id;
    return id;
}


std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

int main(int argc, char* argv[]) {
    printCompilerInfo();
    int deviceCnt;
    HIPCHECK(hipGetDeviceCount(&deviceCnt));
    std::cout << "Number of HIP visible devices is '" << deviceCnt << "'." << std::endl;
    /* extract the following lines from rocm-smi

    GPU[0]              : PCI Bus: 0000:43:00.0
    GPU[1]              : PCI Bus: 0000:23:00.0
    GPU[2]              : PCI Bus: 0000:26:00.0
    GPU[3]              : PCI Bus: 0000:03:00.0
    GPU[4]              : PCI Bus: 0000:E3:00.0
    GPU[5]              : PCI Bus: 0000:C3:00.0
    GPU[6]              : PCI Bus: 0000:C6:00.0
    GPU[7]              : PCI Bus: 0000:83:00.0
    */

    std::string results = exec("rocm-smi --showbus");
    std::stringstream ss(results);
    std::string to;

    std::vector<int> gpuId;
    std::vector<std::string> busId;
    int count = 0;

    while (std::getline(ss, to, '\n')) {
        bool found = false;
        int start = to.find("GPU[");
        if (start == std::string::npos) continue;
        start += 4;
        std::string temp = "";
        while (isdigit(to[start])) temp += to[start++];
        found = true;
        if (found) {
            gpuId.push_back(stoi(temp));
            busId.push_back(to.substr(to.find("PCI Bus: ") + 9));
            std::cout << gpuId[count] << " (rocm-smi GPU ID) ==> "<< busId[count] << " ---> ";
            const char* busIdStr = busId[count].c_str();
            int hipDeviceId;
            if (hipDeviceGetByPCIBusId(&hipDeviceId, busIdStr) != hipSuccess) {
                std::cout << "FAILED PCI Bus ID mapping for HIP ---> ";
            } else std::cout << hipDeviceId << " (HIP Device index) ---> ";
            int64_t id = busIdToInt64(busIdStr);
            std::cout << id << " (PCI Bus ID in INT64)"<< std::endl;

        }
        count++;
    }
}