#rps=('160000' '180000' '200000' '220000' '240000')
#memTTC=('0,1,2,3,4,5,6,7,8,9,20,21,22,23,24,25,26,27,28,29')
#unTTC=('10,11,12,13,14,15,16,17,18,19,30,31,32,33,34,35,36,37,38,39')
#count=('9' '8' '6' '4' '3' '2' '1')
rps=200000
memcore='0,1,2,3,4,5,6,7,8,9,20,21,22,23,24,25,26,27,28,29'
uncore='10,11,12,13,14,15,16,17,18,19,30,31,32,33,34,35,36,37,38,39'
memDir='/home/memcached/memcached/memcached_client'
specDir='/home/spec2006'
libDir='/home/libpfm-4.8.0/perf_examples'
load='mcf'


cd "$memDir"
taskset -c "$memcore" memcached -t 20 -m 10240 -n 550 -u root &
taskset -c  "$uncore" ./loader -a ../twitter_dataset/twitter_dataset_30x -s servers.txt -w 10 -S 1 -D 10240 -j -T 1
sleep 5

taskset -c "$uncore" timeout 300 ./loader -a ../twitter_dataset/twitter_dataset_30x -s servers.txt -g 0.8 -T 1 -c 200 -w 10 -e  -r $rps | tee /home/chr/result/memcached/RPS"$rps"_Mix"$load".txt & 

cd "$libDir"
pid=`pgrep memcached`
timeout 30 ./task -t $pid -e UNHALTED_CORE_CYCLES -e INSTRUCTION_RETIRED -e LLC_MISSES -e MEM_LOAD_UOPS_RETIRED:L2_MISS -e MEM_LOAD_UOPS_RETIRED:L3_MISS -e MEM_LOAD_UOPS_RETIRED:L3_HIT -e MEM_LOAD_UOPS_LLC_MISS_RETIRED:REMOTE_FWD  -e MEM_LOAD_UOPS_LLC_MISS_RETIRED:LOCAL_DRAM -e MEM_LOAD_UOPS_LLC_MISS_RETIRED:REMOTE_DRAM -e MEM_LOAD_UOPS_LLC_MISS_RETIRED:REMOTE_HITM -e CYCLE_ACTIVITY:STALL_CYCLES_L2_PENDING  -e OFFCORE_RESPONSE_0:L3_MISS_LOCAL -e OFFCORE_RESPONSE_0:L3_MISS_REMOTE -e OFFCORE_REQUESTS_OUTSTANDING:DEMAND_DATA_RD_CYCLES -e MINOR-FAULTS -e CPU-MIGRATIONS -e CONTEXT-SWITCHES  1> /home/libpfm-4.8.0/perf_examples/result/Mix"$load"_Under"$rps".txt &


cd "$specDir"
source shrc
y=0
while [ $y -lt 3 ]
do
taskset -c "$memcore" runspec --config=mytest.cfg --size=ref --noreportable --tune=base --iterations=1 "$load" &
y=$(($y+1))
done

sleep 10
pid=`pgrep "$load"`
kill -9 $pid
