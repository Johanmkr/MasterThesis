#!/bin/bash

> seeds.txt

for ((seed=1; seed<=200; seed++)) do
	echo "$seed" >> seeds.txt
done
