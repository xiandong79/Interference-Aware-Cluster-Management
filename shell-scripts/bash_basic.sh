#!/bin/bash
load1='namd'
# load2='leslie3d'

load3='lbm'
# load4='sjeng'

# cd "specDir"
cd /home/spec2006
source shrc
# y=0
# while [ $y -lt 3]
# do
numactl --cpubind=0 --membind=0  runspec --config=mytest.cfg --size=ref --noreportable --tune=base --iteration=1 "$load1" &
# numactl --cpubind=0 --membind=0  runspec --config=mytest.cfg --size=ref --noreportable --tune=base --iteration=1 "$load2" &

numactl --cpubind=1 --membind=1  runspec --config=mytest.cfg --size=ref --noreportable --tune=base --iteration=1 "$load3" &
# numactl --cpubind=1 --membind=1  runspec --config=mytest.cfg --size=ref --noreportable --tune=base --iteration=1 "$load4" &
# y=$(($y+1))
# done

sleep 5

cd ..
# cd "libDir"
cd /home/libpfm-4.8.0/perf_examples
pid1=$(pgrep $load1)
# pid2=$(pgrep $load2)
pid3=$(pgrep $load3)
# pid4=$(pgrep $load4)
echo $load1 $pid1
# echo $load2 $pid2
echo $load3 $pid3
# echo $load4 $pid4
./task -t $pid1 -p -e UNHALTED_CORE_CYCLES -e INSTRUCTION_RETIRED -e LLC_MISSES 1> /home/xiandong-Code-Result/result/"$load1".txt &
./task -t $pid3 -p -e UNHALTED_CORE_CYCLES -e INSTRUCTION_RETIRED -e LLC_MISSES 1> /home/xiandong-Code-Result/result/"$load3".txt &

# ./task -t $pid1 -p -e UNHALTED_CORE_CYCLES -e INSTRUCTION_RETIRED -e LLC_MISSES 1> /home/xiandong-Code-Result/result/"$load1"_under_"$load2".txt &
# ./task -t $pid2 -p -e UNHALTED_CORE_CYCLES -e INSTRUCTION_RETIRED -e LLC_MISSES 1> /home/xiandong-Code-Result/result/"$load2"_under_"$load1".txt &
# ./task -t $pid3 -p -e UNHALTED_CORE_CYCLES -e INSTRUCTION_RETIRED -e LLC_MISSES 1> /home/xiandong-Code-Result/result/"$load3"_under_"$load4".txt &
# ./task -t $pid4 -p -e UNHALTED_CORE_CYCLES -e INSTRUCTION_RETIRED -e LLC_MISSES 1> /home/xiandong-Code-Result/result/"$load4"_under_"$load3".txt &

sleep 100

kill -9 `pgrep base`
kill -9 `pgrep gcc`
kill -9 `pgrep live`
