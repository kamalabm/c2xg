#!/bin/bash
#SBATCH --job-name=c2xg-learn
#SBATCH --account=def-stick           # CHANGE THIS to your PI's account
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=7:00:00               # Short test
#SBATCH --output=c2xg-%j.out
#SBATCH --error=c2xg-%j.err

echo "Job started at $(date)"
echo "Running on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

# Copy data to local node storage (fast SSD)
echo "Copying data to $SLURM_TMPDIR..."
mkdir -p $SLURM_TMPDIR/data/IN
mkdir -p $SLURM_TMPDIR/data/OUT

cp /scratch/kamalabm/claude-c2xg/data/IN/* $SLURM_TMPDIR/data/IN/
cp -r /scratch/kamalabm/claude-c2xg/c2xg $SLURM_TMPDIR/
cp /scratch/kamalabm/claude-c2xg/learn_ubc_args.py $SLURM_TMPDIR/

# Copy model files
echo "Copying model files..."
cp /scratch/kamalabm/claude-c2xg/training_corpus.01.cbow.bin $SLURM_TMPDIR/ 2>/dev/null || echo "Warning: CBOW model not found"
cp /scratch/kamalabm/claude-c2xg/training_corpus.01.sg.bin $SLURM_TMPDIR/ 2>/dev/null || echo "Warning: SG model not found"

echo "Data copied. Disk usage:"
du -sh $SLURM_TMPDIR/*

# Verify model files
echo "Checking for model files:"
ls -lh $SLURM_TMPDIR/*.bin 2>/dev/null || echo "No model files found in $SLURM_TMPDIR"

# Activate conda environment
echo "Activating conda environment..."
source /home/kamalabm/scratch/aall-c2xg/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export PATH=/home/kamalabm/scratch/aall-c2xg/miniconda3/envs/myenv/bin:$PATH

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Run with command-line arguments
echo "Starting C2xG learning..."
cd $SLURM_TMPDIR
python learn_ubc_args.py \
    $SLURM_TMPDIR/data/IN/aall-embeddings-corpus.txt \
    $SLURM_TMPDIR/data/IN/aall-grammar-corpus.txt \
    $SLURM_TMPDIR/data/OUT \
    --nickname "aall_c2xg" \
    --npmi-threshold 0.70 \
    --min-count 5 \
    --max-vocab 50000 \
    --forgetting-rounds 25 \
    --learning-rounds 1 \
    --get-examples

# Copy results back to scratch
echo "Copying results back to scratch..."
cp -r $SLURM_TMPDIR/data/OUT/* /scratch/kamalabm/claude-c2xg/data/OUT/

echo "Job finished at $(date)"
echo "Output files in: /scratch/kamalabm/claude-c2xg/data/OUT/"

# Deactivate conda
conda deactivate
