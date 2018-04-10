#!/bin/bash

python main.py & mypid=$!
# If this script is killed, kill the postproc
trap "kill $mypid 2> /dev/null" EXIT

mname=$(basename $(ps -p $mypid -o comm=))
while kill -0 $mypid 2> /dev/null; do
	chpids=$(pgrep -P $mypid)
	[ ! -z "$chpids" ] && chname=$(basename $(ps -p $chpids -o comm=))
	top -n 1 -b -p$mypid >> "$mname"
	[ ! -z "$chpids" ] && top -n 1 -b -p$chpids >> "$chname"
done 

# Disable the trap on a normal exit
trap - EXIT