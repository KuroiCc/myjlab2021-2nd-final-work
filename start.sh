#! /bin/bash

`nohup poetry run start >> $PROJECTS/myjlab2021-2nd-final-work/app/logs/server.log 2>&1 &` ; echo $!
