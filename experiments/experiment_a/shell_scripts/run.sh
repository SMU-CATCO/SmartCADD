#!/bin/bash
#SBATCH -J testing
#SBATCH -o output/testing_%j.out
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH -G 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00 # maximum runtime 2 days


# Load experiment metadata
experiment_name=$(jq -r '.experiment_name' "${PWD}/shell_scripts/config.json")
package_path=$(jq -r '.package_path' "${PWD}/shell_scripts/config.json")
package_name=$(basename "$package_path")
container_image_path=$(jq -r '.container_image_path' "${PWD}/shell_scripts/config.json")
data_path=$(jq -r '.data_path' "${PWD}/shell_scripts/config.json")
config_file=$(jq -r '.config_file' "${PWD}/shell_scripts/config.json")

chmod +x "${PWD}/shell_scripts/entrypoint.sh"
srun\
    --no-container-entrypoint\
    --container-image "${container_image_path}"\
    --container-mounts="${package_path}"/:/"${package_name}",/"${data_path}":/data\
    --container-workdir /"${package_name}"\
    bash -c "/${PWD}/shell_scripts/entrypoint.sh --config_file '${config_file}' --experiment_name '${experiment_name}' --package_name '${package_name}'" 

# To install a container on superpod, take the string from docker pull and execute 
# enroot command.
# docker pull tensorflow/tensorflow:2.15.0.post1-gpu
# enroot import docker://tensorflow/tensorflow:2.15.0.post1-gpu