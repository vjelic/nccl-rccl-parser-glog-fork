# Device ID Mapping

The device indices on HIP, ROCM-SMI, and RCCL (cudaDev) are not necessarily matching. The only unique information for devices is PCI bus ID. Therefore, this sub-project is meant to create a debugging tool across different software stacks. 



**On ROCM-SMI (--showbus):**
The following outputs are from this tool on a server with 8 MI60 GPUs.
```
======================= ROCm System Management Interface =======================
================================== PCI Bus ID ==================================
GPU[0]          : PCI Bus: 0000:43:00.0
GPU[1]          : PCI Bus: 0000:23:00.0
GPU[2]          : PCI Bus: 0000:26:00.0
GPU[3]          : PCI Bus: 0000:03:00.0
GPU[4]          : PCI Bus: 0000:E3:00.0
GPU[5]          : PCI Bus: 0000:C3:00.0
GPU[6]          : PCI Bus: 0000:C6:00.0
GPU[7]          : PCI Bus: 0000:83:00.0
================================================================================
============================= End of ROCm SMI Log ==============================
```


**Device ID Mapping**

The following outputs are from this tool on a server with 8 MI60 GPUs. We can use PCI bus id to get device id on HIP by using [hipDeviceGetByPCIBusId](https://rocmdocs.amd.com/en/latest/ROCm_API_References/HIP_API/Initialization-and-Version.html?highlight=hipDeviceGetByPCIBusId#hipdevicegetbypcibusid).

```
Number of HIP visible devices is '8'.
0 (rocm-smi GPU ID) ==> 0000:43:00.0 ---> 0 (HIP Device index) ---> 274432 (PCI Bus ID in INT64)
1 (rocm-smi GPU ID) ==> 0000:23:00.0 ---> 1 (HIP Device index) ---> 143360 (PCI Bus ID in INT64)
2 (rocm-smi GPU ID) ==> 0000:26:00.0 ---> 2 (HIP Device index) ---> 155648 (PCI Bus ID in INT64)
3 (rocm-smi GPU ID) ==> 0000:03:00.0 ---> 3 (HIP Device index) ---> 12288 (PCI Bus ID in INT64)
4 (rocm-smi GPU ID) ==> 0000:E3:00.0 ---> 4 (HIP Device index) ---> 929792 (PCI Bus ID in INT64)
5 (rocm-smi GPU ID) ==> 0000:C3:00.0 ---> 5 (HIP Device index) ---> 798720 (PCI Bus ID in INT64)
6 (rocm-smi GPU ID) ==> 0000:C6:00.0 ---> 6 (HIP Device index) ---> 811008 (PCI Bus ID in INT64)
7 (rocm-smi GPU ID) ==> 0000:83:00.0 ---> 7 (HIP Device index) ---> 536576 (PCI Bus ID in INT64)
```

As the results shown above, we confirmed that the device indices on HIP and ROCM-SMI are identical when there is no HIP_VISIBLE_DEVICES used.


When HIP_VISIBLE_DEVICES is used, the indices will not match for HIP and ROCM-SMI. The following outputs are from
```
HIP_VISIBLE_DEVICES=0,3,7 ./hip_rocm_smi_mapping
```
```
Number of HIP visible devices is '3'.
0 (rocm-smi GPU ID) ==> 0000:43:00.0 ---> 0 (HIP Device index) ---> 274432 (PCI Bus ID in INT64)
1 (rocm-smi GPU ID) ==> 0000:23:00.0 ---> FAILED PCI Bus ID ( 0000:23:00.0) mapping for HIP ---> 143360 (PCI Bus ID in INT64)
2 (rocm-smi GPU ID) ==> 0000:26:00.0 ---> FAILED PCI Bus ID ( 0000:26:00.0) mapping for HIP ---> 155648 (PCI Bus ID in INT64)
3 (rocm-smi GPU ID) ==> 0000:03:00.0 ---> 1 (HIP Device index) ---> 12288 (PCI Bus ID in INT64)
4 (rocm-smi GPU ID) ==> 0000:E3:00.0 ---> FAILED PCI Bus ID ( 0000:E3:00.0) mapping for HIP ---> 929792 (PCI Bus ID in INT64)
5 (rocm-smi GPU ID) ==> 0000:C3:00.0 ---> FAILED PCI Bus ID ( 0000:C3:00.0) mapping for HIP ---> 798720 (PCI Bus ID in INT64)
6 (rocm-smi GPU ID) ==> 0000:C6:00.0 ---> FAILED PCI Bus ID ( 0000:C6:00.0) mapping for HIP ---> 811008 (PCI Bus ID in INT64)
7 (rocm-smi GPU ID) ==> 0000:83:00.0 ---> 2 (HIP Device index) ---> 536576 (PCI Bus ID in INT64)
```