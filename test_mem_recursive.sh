#!/bin/bash

pidtree() {
  echo -n $1 " "
  for _child in $(ps -o pid --no-headers --ppid $1); do
    echo -n $_child `pidtree $_child` " "
  done
}

python test_mem.py & mypid=$!
# If this script is killed, kill the postproc
trap "kill $mypid 2> /dev/null" EXIT

mname=$(basename $(ps -p $mypid -o comm=))
while kill -0 $mypid 2> /dev/null; do
        ps -o pid=,tty=,stat=,time=,rss=,trs=,cmd= f `pidtree $mypid` >> "$1$mname"
done

# Disable the trap on a normal exit
trap - EXIT
