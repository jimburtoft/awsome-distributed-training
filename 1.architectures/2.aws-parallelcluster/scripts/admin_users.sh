#!/bin/bash
# ParallelCluster post-install script for taking all users from args and setting them as Slurm Admin
# AdminLevel=<level> Admin level of user. Valid levels are None, Operator, and Admin. 
# Requires an authentication DB (sssd, ldap, or nis, etc.) to get those users on the OS beforehand 
# Requires 'accounting_storage/slurmdbd'
# https://slurm.schedmd.com/sacctmgr.html
# https://docs.aws.amazon.com/parallelcluster/latest/ug/Scheduling-v3.html#yaml-Scheduling-SlurmQueues-CustomActions-OnNodeConfigured

ADMIN_USERS="${*}"
export PATH=${PATH}:/opt/slurm/bin


for u in $ADMIN_USERS ;do
  sacctmgr -i add user name=${u} DefaultAccount=pcdefault AdminLevel=Admin
done
sacctmgr show users
sacctmgr list associations

