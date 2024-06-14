#!/bin/bash

echo -e "Ejecutando version_base.c" > output.txt
gcc -o version_base version_base.c
./version_base >> output.txt

for i in {0..3}; do
  echo -e "\nEjecutando version_base.c con -O$i" >> output.txt
  gcc -O$i -o version_base_$i version_base.c
  ./version_base_$i >> output.txt
done

for i in {1..5}; do
  echo -e "\t===== Ejecutando version_$i.c =====" >> output.txt
  echo -e "Ejecutando version_$i.c off" >> output.txt
  gcc -o version_$i version_$i.c
  ./version_$i >> output.txt
  for j in {0..3}; do
    echo -e "\nEjecutando version_$i.c con -O$j" >> output.txt
    gcc -O$j -o version_${i}_$j version_$i.c
    ./version_${i}_$j >> output.txt
  done
  echo -e "\n" >> output.txt
done