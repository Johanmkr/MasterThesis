#!/bin/bash

> seeds.txt

for ((seed=1; seed<=1000; seed++)) do
	echo "$seed" >> seeds.txt
done
