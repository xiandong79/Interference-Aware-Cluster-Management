#!/bin/sh
# benchmark=(gobmk mcf milc hmmer)
# loadarray=(bzip2 bwaves mcf milc leslie3d namd gobmk lbm libquantum sjeng)

# this script is used to run 7 different ibench tests in a row which saves a great time.

cnt=0

while [ $cnt -lt 7 ]
do
    	./ibench.sh $cnt
    	sleep 5
	./kill_thread.sh
	sleep 2 
    	cnt=`expr $cnt + 1` 
done

