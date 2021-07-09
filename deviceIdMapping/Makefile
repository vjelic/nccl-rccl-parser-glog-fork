HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
        HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=hip_rocm_smi_mapping

all: $(EXE)

$(EXE): hip_rocm_smi_mapping.cpp
        $(HIPCC) $(EXE).cpp -o $@

clean:
        rm -f *.o $(EXE)