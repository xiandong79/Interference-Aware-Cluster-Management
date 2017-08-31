#!/bin/sh
#benchmark=(gobmk mcf milc hmmer)

loadarray="bzip2 bwaves mcf milc leslie3d namd gobmk lbm libquantum sjeng"

for i in $loadarray; do
	cnt=0
        echo $i
	while [ $cnt -lt 7 ]
	do
    		./ibench.sh $cnt
    		sleep 5
		./kill_thread.sh
		sleep 2 
    	cnt=`expr $cnt + 1` 
	done
done

