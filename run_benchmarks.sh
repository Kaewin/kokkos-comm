#!/bin/bash

echo "=== RMA Performance Comparison ==="
echo "Operation,MessageSize,Processes,AvgTime(seconds)"
echo ""

# Test with 2 processes for basic tests
for size in 1 8 64 512 4096 32768 262144 1048576; do
    echo "Testing size: $size elements"
    mpirun -n 2 ./build/unit_tests/benchmark_lock_unlock --size $size --iterations 100 --gtest_brief=1
done

echo ""
echo "=== Testing with 3 processes (enables Shared Get) ==="
for size in 1 8 64 512 4096 32768 262144; do
    echo "Testing size: $size elements"
    mpirun -n 3 ./build/unit_tests/benchmark_lock_unlock --size $size --iterations 100 --gtest_brief=1
done