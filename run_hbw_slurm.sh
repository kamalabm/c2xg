#!/bin/bash
  #SBATCH --job-name=c2xg-prod
  #SBATCH --account=def-yourpi
  #SBATCH --time=48:00:00        # 2 days
  #SBATCH --cpus-per-task=40     # More for larger matrices
  #SBATCH --mem=200G             # Scaled up from 82GB
  #SBATCH --output=c2xg-%j.out
  #SBATCH --error=c2xg-%j.err

  echo "Job started at $(date)"
  echo "Running on node: $HOSTNAME"
  echo "Job ID: $SLURM_JOB_ID"

  # Copy data to local node storage (fast SSD)
  echo "Copying data to $SLURM_TMPDIR..."
  mkdir -p $SLURM_TMPDIR/data/IN
  mkdir -p $SLURM_TMPDIR/data/OUT

  # Copy production corpus (THIS WILL TAKE LONGER)
  cp /scratch/kamalabm/claude-c2xg/data/IN/production-embeddings-corpus.txt $SLURM_TMPDIR/data/IN/
  cp /scratch/kamalabm/claude-c2xg/data/IN/production-grammar-corpus.txt $SLURM_TMPDIR/data/IN/
  cp -r /scratch/kamalabm/claude-c2xg/c2xg $SLURM_TMPDIR/
  cp /scratch/kamalabm/claude-c2xg/learn_ubc_args.py $SLURM_TMPDIR/

  # Copy model files
  echo "Copying model files..."
  cp /scratch/kamalabm/claude-c2xg/production_corpus.cbow.bin $SLURM_TMPDIR/ || echo "Warning: CBOW model not found"
  cp /scratch/kamalabm/claude-c2xg/production_corpus.sg.bin $SLURM_TMPDIR/ || echo "Warning: SG model not found"

  echo "Data copied. Disk usage:"
  du -sh $SLURM_TMPDIR/*

  # Activate conda environment
  echo "Activating conda environment..."
  source /home/kamalabm/scratch/aall-c2xg/miniconda3/etc/profile.d/conda.sh
  conda activate myenv
  export PATH=/home/kamalabm/scratch/aall-c2xg/miniconda3/envs/myenv/bin:$PATH

  # Verify environment
  echo "Python: $(which python)"
  echo "Python version: $(python --version)"

  # Run with production settings
  echo "Starting C2xG learning..."
  cd $SLURM_TMPDIR
  python learn_ubc_args.py \
      $SLURM_TMPDIR/data/IN/production-embeddings-corpus.txt \
      $SLURM_TMPDIR/data/IN/production-grammar-corpus.txt \
      $SLURM_TMPDIR/data/OUT \
      --nickname "production_c2xg" \
      --npmi-threshold 0.70 \
      --min-count 5 \
      --max-vocab 80000 \
      --forgetting-rounds 30 \
      --learning-rounds 1 \
      --get-examples

  # Copy results back to scratch
  echo "Copying results back to scratch..."
  cp -r $SLURM_TMPDIR/data/OUT/* /scratch/kamalabm/claude-c2xg/data/OUT/

  echo "Job finished at $(date)"
  echo "Output files in: /scratch/kamalabm/claude-c2xg/data/OUT/"

  conda deactivate

