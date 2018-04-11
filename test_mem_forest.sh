#!/bin/bash

python test_mem.py & mypid=$!

# If this script is killed, kill the postproc
trap "kill $mypid 2> /dev/null" EXIT

mname=$(basename $(ps -p $mypid -o comm=))
while kill -0 $mypid 2> /dev/null; do
        ps --forest -o pid=,tty=,stat=,time=,rss=,trs=,cmd= -g $(ps -o sid= -p $mypid) >> "$1$mname"
        #ps --forest -o pid=,tty=,stat=,time=,rss=,trs=,cmd= -g  $mypid >> "$1$mname"
done

# Disable the trap on a normal exit
trap - EXIT