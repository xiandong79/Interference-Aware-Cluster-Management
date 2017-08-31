#!/bin/bash
cd /home/spec2006

loadarray=( bzip2 bwaves mcf milc leslie3d namd gobmk lbm libquantum sjeng )
  
for i in ${loadarray[@]}; do  
	
	for j in ${loadarray[@]}; do  

	load1=$i
	load2=$j

	if [ "$load1" = "$load2" ]
	then

		numactl --cpubind=1 --membind=1  runspec --config=mytest.cfg --size=ref --noreportable --tune=base --iteration=1 "$load2" &

		sleep 2

		pid2=$(pgrep $load2)

		./home/libpfm-4.8.0/perf_examples/task -t $pid2 -p -e UNHALTED_CORE_CYCLES -e INSTRUCTION_RETIRED -e LLC_MISSES 1> /home/xiandong-Code-Result/result-for-loop"$load2".txt &

	else 

		numactl --cpubind=0 --membind=0  runspec --config=mytest.cfg --size=ref --noreportable --tune=base --iteration=1 "$load1" &
		numactl --cpubind=0 --membind=0  runspec --config=mytest.cfg --size=ref --noreportable --tune=base --iteration=1 "$load2" &

		sleep 2

		pid1=$(pgrep $load1)
		pid2=$(pgrep $load2)
		echo $load1 $pid1
		echo $load2 $pid2

		./home/libpfm-4.8.0/perf_examples/task -t $pid1 -p -e UNHALTED_CORE_CYCLES -e INSTRUCTION_RETIRED -e LLC_MISSES 1> /home/xiandong-Code-Result/result-for-loop/"$load1"_under_"$load2".txt &
		./home/libpfm-4.8.0/perf_examples/task -t $pid2 -p -e UNHALTED_CORE_CYCLES -e INSTRUCTION_RETIRED -e LLC_MISSES 1> /home/xiandong-Code-Result/result-for-loop/"$load2"_under_"$load1".txt &

	fi
	
		
	kill -9 `pgrep base`
	kill -9 `pgrep base`
	kill -9 `pgrep base`
	
	done
done
