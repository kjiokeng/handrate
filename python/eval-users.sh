#!/bin/bash

users=(
	DEMO
	USER
	USER1
	USER2
	USER3
	USER4
	USER5
	USER6
	USER7
	USER8
	USER9
	USER100
	USER101
	USER102
	USER103
	USER104
	USER105
	USER106
	USER107
	USER108
	)

for user in ${users[*]}; do
	#statements
	python3.5 handratenn.rnn.eval.py -m handrate.rnn.conv.users.${user}.3s.10ones.128units.h5 -d ../matlab/datasets/users/handrate-dataset-raw-${user}-3s-10ones-pca.mat -n 3 -u 128 -D 65 -v 0.25 -o 100 -s train
	echo "================== end for user ${user} =================="
done