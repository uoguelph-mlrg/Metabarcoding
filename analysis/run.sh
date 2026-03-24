#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=metabarcoding_analysis
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
#SBATCH --qos=normal
#SBATCH --open-mode=append

# Load necessary modules
module load python/3.12 cuda/12.6 arrow/21.0.0 opencv/4.12.0 

# Activate virtual environment
source ~/.bashrc
source ~/barcode/bin/activate

# Interpolated latent
cd interpolated_latent/V4
python interpolated_latent.py

# Location embeddings
cd ../../location_embedding
python location_embedding.py

cd ..
python visualize_results.py --results_dir interpolated_latent/V4/results/model_comparison_results.pkl --output_dir figures/interpolated_latent
python visualize_results.py --results_dir location_embedding/results/location_embeddings_comparison.pkl --output_dir figures/location_embeddings