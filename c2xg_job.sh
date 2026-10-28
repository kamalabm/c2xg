#!/bin/bash
#SBATCH --job-name=c2xg-learn
#SBATCH --account=def-yourpi
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=25
#SBATCH --mem=200G
#SBATCH --output=c2xg-%j.out
#SBATCH --error=c2xg-%j.err

echo "Job started at $(date)"
echo "Running on node: $HOSTNAME"

# Copy data to local node storage
echo "Copying data to $SLURM_TMPDIR..."
mkdir -p $SLURM_TMPDIR/data/IN
mkdir -p $SLURM_TMPDIR/data/OUT

cp /scratch/kamalabm/claude-c2xg/data/IN/* $SLURM_TMPDIR/data/IN/
cp -r /scratch/kamalabm/claude-c2xg/c2xg $SLURM_TMPDIR/
cp /scratch/kamalabm/claude-c2xg/learn_ubc.py $SLURM_TMPDIR/

echo "Data copied. Disk usage:"
du -sh $SLURM_TMPDIR/*

# Activate conda
source /home/kamalabm/scratch/aall-c2xg/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export PATH=/home/kamalabm/scratch/aall-c2xg/miniconda3/envs/myenv/bin:$PATH

# Run from local storage
cd $SLURM_TMPDIR
echo "Running from: $(pwd)"
python learn_ubc.py $SLURM_TMPDIR/data/IN/aall-embeddings-corpus.txt \
                     $SLURM_TMPDIR/data/IN/aall-grammar-corpus.txt \
                     $SLURM_TMPDIR/data/OUT/

# Copy results back
echo "Copying results back..."
cp -r $SLURM_TMPDIR/data/OUT/* /scratch/kamalabm/claude-c2xg/data/OUT/

echo "Job finished at $(date)"
conda deactivate

