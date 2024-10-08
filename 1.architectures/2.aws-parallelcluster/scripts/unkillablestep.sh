#!/bin/bash
# ParallelCluster post-install script for Compute nodes (OnNodeConfigured/Script) to add the Slurm UnkillableStep feature to reboot a node with a zombie process
# setting in slurm.conf the variables UnkillableStepProgram and UnkillableStepTimeout
# creating a script to reboot a node
# https://slurm.schedmd.com/slurm.conf.html
# https://docs.aws.amazon.com/parallelcluster/latest/ug/Scheduling-v3.html#yaml-Scheduling-SlurmQueues-CustomActions-OnNodeConfigured

# UnkillableStepProgram:
# If the processes in a job step are determined to be unkillable for a period of time specified by the UnkillableStepTimeout variable,
# the program specified by UnkillableStepProgram will be executed. By default no program is run.

# UnkillableStepTimeout:
# The length of time, in seconds, that Slurm will wait before deciding that processes in a job step are unkillable (after they have been signaled with SIGKILL) and
# execute UnkillableStepProgram. The default timeout value is 60 seconds or five times the value of MessageTimeout, whichever is greater.
# If exceeded, the compute node will be drained to prevent future jobs from being scheduled on the node.
# NOTE: Ensure that UnkillableStepTimeout is at least 5 times larger than MessageTimeout, otherwise it can lead to unexpected draining of nodes. 


export PATH=${PATH}:/opt/slurm/bin
export SLURMDIR="/opt/slurm/etc"
export SLURMCONF="${SLURMDIR}/slurm.conf"
export LOCALSBIN="/usr/local/sbin"
export SLURMUNKILLABLE="${LOCALSBIN}/unkillablestep.sh"

sed -i $SLURMCONF -e '/UnkillableStepProgram=/d' -e '/UnkillableStepTimeout=/d' 
cat >> $SLURMCONF << EOF
UnkillableStepProgram=${SLURMUNKILLABLE}
UnkillableStepTimeout=60
EOF

mkdir -p "${LOCALSBIN}"
cat > $SLURMUNKILLABLE << \EOF
#!/bin/bash

scontrol reboot ASAP nextstate=RESUME reason="unkillablestep-reboot" $(hostname) 

exit
EOF

# restart slurm
scontrol reconfig
# systemctl restart slurmctld
systemctl status slurmctld

exit



