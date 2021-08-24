#!/bin/bash
sudo apt-get install -y gawk
sudo apt-get install -y graphviz
pip install pandas
pip install networkx
cd deviceIdMapping; make
./hip_rocm_smi_mapping > busId_HIP_map.txt

read -p "===== Which system are you using? (Please either type rocm or cuda.) =====" SYS

if [ "$SYS" = "rocm" ]
then
    cd ../rccl-tests/;make
elif [ "$SYS" = "cuda" ]
then
    cd ../nccl-tests/;make
else
    echo "Please run this bash script again with correct system specified."
fi