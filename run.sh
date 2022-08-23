#!/bin/bash

# Usage: ./run.sh --device cuda:0 --train --bench --demo --test
set -e

# Default:
device="cuda:0"
train=false
bench=false
demo=false
test=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --device) device="$2"; shift ;;
        --train)  train=true ;;
        --bench)  bench=true ;;
        --demo)   demo=true ;;
        --test)   test=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

run_start=`date +%s`
echo "Start (device=$device, train=$train, bench=$bench, demo=$demo, test=$test)"

echo "========"

echo "Delete"
$demo  && ( set -x; rm -rf "saved/demo_*" )
$test  && ( set -x; rm -rf "saved/test_*" )

echo "========"

echo "Standard Cliff Walking (Solution based on Q table)"
$train && ( set -x; python standard_qtable.py )
$demo  && ( set -x; python demo_standard_qtable.py --run )

echo "========"

echo "Advanced Cliff Walking (Solution based on DQN)"
$train && ( set -x; python advanced_dqn.py --device $device )
$bench && ( set -x; python bench_advanced_dqn.py --device $device )
$demo  && ( set -x; python demo_advanced_dqn.py --device $device --run )

echo "========"

echo "Advanced Cliff Walking with an indefinite number of cliffs (Solution based on DQN)"
$train && ( set -x; python advanced_dqn.py --device $device --rand )
$bench && ( set -x; python bench_advanced_dqn.py --device $device --rand )
$demo  && ( set -x; python demo_advanced_dqn.py --device $device --run --rand )

echo "========"

echo "Other"
$test  && ( set -x; python test_env_check.py )
$demo  && ( set -x; python demo_other.py )

echo "========"

run_end=`date +%s`
run_time=$((run_end-run_start))

echo "Done (run_time=$run_time seconds)"
