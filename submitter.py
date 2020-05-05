import json
import subprocess
from datetime import datetime

# STEPS:
# * load run variables
# * write template variables sh file
# * launch monitoring job for getting parseable results
# This is better done as a general job in the host.
# nvidia-smi --format=csv --query-gpu=timestamp,index,utilization.gpu,utilization.memory,pstate,temperature.gpu,memory.used,memory.free -l 5 -f /pylon5/pscstaff/julian/mlperf/results/2020-05-01/general.gpu.log &
# pidstat -C "python|singularity" -h -d -r -u -U julian 5 >> /pylon5/pscstaff/julian/mlperf/results/2020-05-01/julian.cpu.log &
# * launch sbatch job


run_commands = []
runs_per_config = 5
run_cofigurations = [
    {"nodes": 1, "wall_time": "00:10:00", "n_gpu": 1, "cpu_cores": 1, "n_cpu": 1},
    {"nodes": 1, "wall_time": "00:10:00", "n_gpu": 2, "cpu_cores": 2, "n_cpu": 1},
    {"nodes": 1, "wall_time": "00:10:00", "n_gpu": 4, "cpu_cores": 4, "n_cpu": 1},
    {"nodes": 1, "wall_time": "00:10:00", "n_gpu": 8, "cpu_cores": 8, "n_cpu": 1},
    {"nodes": 1, "wall_time": "00:10:00", "n_gpu": 16, "cpu_cores": 16, "n_cpu": 1},
]

for run_cofiguration in run_cofigurations:
    nodes, wall_time, n_gpu, cpu_cores, n_cpu = run_cofiguration.get('nodes'), \
                                                run_cofiguration.get('wall_time'), \
                                                run_cofiguration.get('n_gpu'), \
                                                run_cofiguration.get('cpu_cores'), \
                                                run_cofiguration.get('n_cpu')

    file_name = "job_{nodes}Nodes_{n_gpu}GPUs_{n_cpu}CPUs".format(nodes=nodes, n_gpu=n_gpu, n_cpu=n_cpu)
    batch_job = """#!/bin/bash
#SBATCH -N {nodes}
#SBATCH --partition=GPU-AI
#SBATCH --gres=gpu:volta32:{n_gpu}
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=EGRESS
#SBATCH --reservation=dgx2
#SBATCH --time={wall_time}
#SBATCH --exclusive

cd $LOCAL/
rsync -a $SCRATCH/mlperf/production*.sif .
rsync -a $SCRATCH/mlperf/training_results_v0.5* .
rsync -a $SCRATCH/mlperf/data* .
cd $LOCAL/training_results_v0.5/v0.5.0/nvidia/submission/code/recommendation/pytorch/

echo '#!/bin/bash
echo NODES: '{nodes}' WALLTIME: '{wall_time}' NGPU: '{n_gpu}' CORES: '{cpu_cores}' CPUS '{n_cpu}'
## DL params
EXTRA_PARAMS=( )
## System run parms
DGXNNODES='{nodes}'
DGXSYSTEM=DGX2
WALLTIME='{wall_time}'
## System config params
DGXNGPU='{n_gpu}'
DGXSOCKETCORES='{cpu_cores}'
DGXNSOCKET='{n_cpu}'
DGXHT=1         # HT is on is 2, HT off is 1
DGXIBDEVICES=""' > config_interact.sh

source /etc/profile.d/modules.sh
module load singularity
singularity exec --nv -B $LOCAL/data:/data $LOCAL/production_root_pytorch_18.11-py3.sif ./run_and_time.sh 
""".format(nodes=nodes, wall_time=wall_time, n_gpu=n_gpu, cpu_cores=cpu_cores, n_cpu=n_cpu, file_name=file_name)

    with open(file_name + ".sbatch.sh", "w") as file:
        file.write(batch_job)

    run_commands.append("/opt/packages/slurm/default/bin/sbatch {file_name}.sbatch.sh".format(file_name=file_name))

results_dict = {}
for run_command in run_commands:
    for run_number in range(runs_per_config):
        # process_handle = subprocess.run(run_command.split(" "), capture_output=True)
        # run_output = process_handle.stdout.decode("utf-8")[:-1].split("\n")
        print(run_command)

        # file_name = run_command.split(" ")[1]
        #
        # with open(file_name, "w+") as file:
        #     content = json.dumps(results_dict)
        #     file.write(content)

print('pidstat -C "python|singularity" -h -d -r -u -U julian 5 >> '
      '/pylon5/pscstaff/julian/mlperf/results/2020-05-01/julian.cpu.log &')
