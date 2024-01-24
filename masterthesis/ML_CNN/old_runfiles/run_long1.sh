#!/bin/bash

python3 long1_500_1000.py
python3 long1_1000_1500.py

for i in {0..3} 
do
    python3 long1_0_500.py
    python3 long1_500_1000.py
    python3 long1_1000_1500.py
done
