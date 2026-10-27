#!/bin/bash
# Usage: PBS_ACCOUNT=<project> [SETUP_SH=<env script>] bash pbs/submit.sh <name> <walltime> <payload.sh> [queue]
# Runs the payload in a one-node job with QA (this directory) and RUN (runs/<name>_<time>/) exported.
set -euo pipefail
NAME=${1:?name}; WALL=${2:?walltime}; SCRIPT=$(readlink -f "${3:?script}"); QUEUE=${4:-capacity}
QA=${QA:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}
ACCOUNT=${PBS_ACCOUNT:?set PBS_ACCOUNT}
ENVLINE=${SETUP_SH:+source $SETUP_SH}; ENVLINE=${ENVLINE:-module load frameworks}
RUN="$QA/runs/${NAME}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RUN"; cp "$SCRIPT" "$RUN/payload.sh"
cat > "$RUN/job.pbs" <<PBS
#!/bin/bash
#PBS -l select=1,place=scatter,walltime=$WALL
#PBS -l filesystems=flare
#PBS -q $QUEUE
#PBS -A $ACCOUNT
#PBS -j oe
#PBS -o $RUN/job.log
#PBS -N $NAME
$ENVLINE
module load xpu-smi 2>/dev/null || true
export QA=$QA RUN=$RUN
cd $QA
bash $RUN/payload.sh
PBS
qsub "$RUN/job.pbs"
