#!/bin/bash
# create prolog and epiog to manage multitenancy in ParallelCluster
# PrologFlags=Alloc will force the script to be executed at job allocation
# https://docs.aws.amazon.com/parallelcluster/latest/ug/slurm-prolog-epilog-v3.html
# https://slurm.schedmd.com/prolog_epilog.html

export SLURMDIR="/opt/slurm/etc"
export SLURMEPIDIR="${SLURMDIR}/scripts/epilog.d"
export SLURMPRODIR="${SLURMDIR}/scripts/prolog.d"

cat > "${SLURMPRODIR}/dynamic-fsxl-mount.sh" << \EOF
#!/bin/bash
# slurmd root user
# Prolog for Slurm to mount the dynamic FSxL associated with the user group
# 1. Identify Job's team
# 2. Select Right FSx from a logic with name convention, FSxL tag, etc.
# 3. Mount the FSxL
# SLURM_JOB_UID User ID of the job's owner.
# SLURM_JOB_USER User name of the job's owner.
# SLURM_JOB_GID Group ID of the job's owner.
# id -nG $USER

grp="$(getent group ${SLURM_JOB_GID} | cut -d':' -f1)" # example

case ${grp} in
  team1)
    fsxl=""
  ;;
  team2)
    fsxl=""
  ;;
  *)
    echo "ERROR: no group found"
    fsxl=""
  ;;
esac

mount ${fsxL} # if /etc/fstab is filled already
# mount -t lustre -o relatime,flock file_system_dns_name@tcp:/mountname /fsx

exit
EOF

cat > "${SLURMEPIDIR}/dynamic-fsxl-umount.sh" << \EOF
#!/bin/bash
# slurmd root user
# Epilog for Slurm to only keep the static FSxL and umount all other FSxL's:
# 1. SYNC the FSxL
# 2. list all the dynamix FSxL, from a logic with name convention, FSxL tag, etc.
# 3. unmount all of them

FSXL_STATIC="/fsx" # string to recognize the static FSxL to keep mounted

for fsxl in $(mount -t lustre | awk '{print $3}' | grep -v "${FSXL_STATIC}" ) ;do
  sync -f "${fsxl}"
  lsof "${fsxl}" | awk '{print $2}' | xargs kill -9 # kill all processes which could use the ${fsxl}
  umount "${fsxl}" # umount could fail
  umount -f "${fsxl}" # Force the unmount
  umount -l "${fsxl}" # Detach  the  filesystem
done

exit
EOF


# restart slurm
scontrol reconfig
# systemctl restart slurmctld
systemctl status slurmctld


