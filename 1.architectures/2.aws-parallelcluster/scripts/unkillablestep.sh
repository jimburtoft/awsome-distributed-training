#!/bin/bash


export SLURMDIR="/opt/slurm/etc"
export SLURMCONF="${SLURMDIR}/slurm.conf"
export SLURMUNKILLABLE="/usr/local/sbin/unkillablestep.sh" 

sed -i $SLURMCONF -e '/UnkillableStepProgram=/d' -e '/UnkillableStepTimeout=/d' 

cat >> $SLURMCONF << EOF
UnkillableStepProgram=${SLURMUNKILLABLE}
UnkillableStepTimeout=60 # WARNING about supportpushlogs.sh timeout for ComputeFleet
EOF

# restart slurm
  systemctl restart slurmctld
  systemctl status slurmctld


# scontrol reboot ASAP nextstate=RESUME reason="hung_proc_log-n-reboot" $(hostname) 


