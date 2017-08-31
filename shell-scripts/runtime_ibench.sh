#!/bin/bash
# In this script, each workload (i.e. bzip2) will run its full process and full runtime. 

load1='libquantum'

loadarray="bzip2 bwaves mcf milc leslie3d namd gobmk lbm libquantum sjeng"

cd /home/xiandong-Code-Result/src/l1d
echo 3 > /proc/sys/vm/drop_caches
case $1 in
    0) 	taskset -c 3 ./cpu 2000  & 
	sleep 5
	bench_pid=$(pgrep cpu)
	echo  $1 $bench_pid 
    ;;
    1)  taskset -c 3 ./l1d 2000 &  
	sleep 5
	bench_pid=$(pgrep l1d)
	echo  $1 $bench_pid 
    ;;
    2)  taskset -c 3 ./l1i 2000 1 &
	sleep 5
	bench_pid=$(pgrep l1i) 
	echo $1 $bench_pid
    ;;
    3)  taskset -c 3 ./l2d 2000 & 
	sleep 5
	bench_pid=$(pgrep l2d) 
	echo $1 $bench_pid
    ;;
    4)  numactl --cpubind=0 --membind=0 ./l3d 2000 & 
	sleep 5 
	bench_pid=$(pgrep l3d)
	echo $1 $bench_pid 
    ;;
    5)  numactl --cpubind=0 --membind=0 ./membw 2000  & 
	sleep 5 
	bench_pid=$(pgrep membw) 
	echo $1 $bench_pid
    ;;
    6)  numactl --cpubind=0 --membind=0 ./memcap 2000 & 
	sleep 5 
	bench_pid=$(pgrep memcap) 
	echo $1 $bench_pid
    ;;
esac


sleep 3


cd /home/spec2006
numactl --cpubind=0 --membind=0  runspec --config=mytest.cfg --size=ref --noreportable --tune=base --iteration=1 "$load1" &

sleep 5

cd ..
cd /home/libpfm-4.8.0/perf_examples
pid1=$(pgrep $load1)
echo $load1 $pid1
echo $1 $bench_pid

./task -t $pid1 -p -e UNHALTED_CORE_CYCLES -e INSTRUCTION_RETIRED -e LLC_MISSES 1> /home/xiandong-Code-Result/result/"$load1"_under_"$1".txt &
./task -t $bench_pid -p -e UNHALTED_CORE_CYCLES -e INSTRUCTION_RETIRED -e LLC_MISSES 1> /home/xiandong-Code-Result/result/"$1"_under_"$load1".txt &

# sleep 100

# kill -9 `pgrep base`
# kill -9 `pgrep gcc`
# kill -9 `pgrep live`

specperl_pid= pidof specperl
while $specperl_pid   
do
  echo ""
done

 if [ ! $specperl_pid ]
    then
        case $1 in
        0) ps aux | grep cpu; killall -9 cpu
        ;;
        1) ps aux | grep l1d; killall -9 l1d
        ;;
        2) ps aux | grep l1i; killall -9 l1i
        ;;
        3) ps aux | grep l2d; killall -9 l2d
        ;;
        4) ps aux | grep l3d; killall -9 l3d
        ;;
        5) ps aux | grep membw; killall -9 membw
        ;;
        6) ps aux | grep memcap; killall -9 memcap
        ;;
        esac
 fi

echo "=This test is over !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
