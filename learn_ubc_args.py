#!/usr/bin/env python
"""
C2xG Grammar Learning Script with Command-Line Arguments
"""
import argparse
from pathlib import Path
import os, sys

# Ensure we import the package from the repo checkout (c2xg/) not the parent folder shadow
ROOT = Path(__file__).resolve().parent
PKG_ROOT = ROOT / "c2xg"
if PKG_ROOT.exists():
    sys.path.insert(0, str(PKG_ROOT))

from c2xg import C2xG


def main():
    parser = argparse.ArgumentParser(
        description='Learn C2xG grammar from corpus data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('embeddings_corpus', type=str,
                        help='Path to embeddings corpus file')
    parser.add_argument('grammar_corpus', type=str,
                        help='Path to grammar corpus file')
    parser.add_argument('output_dir', type=str,
                        help='Output directory for results')

    # C2xG initialization parameters
    parser.add_argument('--nickname', type=str, default='ubc_novels',
                        help='Nickname for this run')
    parser.add_argument('--language', type=str, default='eng',
                        help='Language code')
    parser.add_argument('--max-words', type=int, default=10000000,
                        help='Maximum number of words to process')
    parser.add_argument('--max-sentence-length', type=int, default=100,
                        help='Maximum sentence length')
    parser.add_argument('--cbow-file', type=str, default='training_corpus.01.cbow.bin',
                        help='CBOW model filename')
    parser.add_argument('--sg-file', type=str, default='training_corpus.01.sg.bin',
                        help='Skip-gram model filename')
    parser.add_argument('--normalization', action='store_true', default=True,
                        help='Enable normalization')
    parser.add_argument('--no-normalization', dest='normalization', action='store_false',
                        help='Disable normalization')

    # Learning parameters
    parser.add_argument('--npmi-threshold', type=float, default=0.70,
                        help='NPMI threshold for associations')
    parser.add_argument('--starting-index', type=int, default=1,
                        help='Starting index for learning')
    parser.add_argument('--min-count', type=int, default=5,
                        help='Minimum word count threshold')
    parser.add_argument('--max-vocab', type=int, default=50000,
                        help='Maximum vocabulary size')
    parser.add_argument('--cbow-range', action='store_true', default=False,
                        help='Use CBOW range')
    parser.add_argument('--sg-range', action='store_true', default=False,
                        help='Use skip-gram range')
    parser.add_argument('--get-examples', action='store_true', default=True,
                        help='Generate examples')
    parser.add_argument('--no-examples', dest='get_examples', action='store_false',
                        help='Do not generate examples')
    parser.add_argument('--increments', type=int, default=25000,
                        help='Processing increments')
    parser.add_argument('--learning-rounds', type=int, default=1,
                        help='Number of learning rounds')
    parser.add_argument('--forgetting-rounds', type=int, default=25,
                        help='Number of forgetting rounds')
    parser.add_argument('--cluster-only', action='store_true', default=False,
                        help='Run clustering only')

    args = parser.parse_args()

    # Verify paths
    if not os.path.exists(args.embeddings_corpus):
        print(f"Error: Embeddings corpus not found: {args.embeddings_corpus}")
        sys.exit(1)

    if not os.path.exists(args.grammar_corpus):
        print(f"Error: Grammar corpus not found: {args.grammar_corpus}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("C2xG Grammar Learning")
    print("=" * 70)
    print(f"Embeddings corpus: {args.embeddings_corpus}")
    print(f"Grammar corpus:    {args.grammar_corpus}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Nickname:          {args.nickname}")
    print(f"Language:          {args.language}")
    print("=" * 70)

    # Initialize C2xG with absolute paths for model files
    cbow_path = os.path.abspath(args.cbow_file) if args.cbow_file else None
    sg_path = os.path.abspath(args.sg_file) if args.sg_file else None

    print(f"CBOW model path: {cbow_path}")
    print(f"SG model path: {sg_path}")

    cxg = C2xG(
        nickname=args.nickname,
        language=args.language,
        max_words=args.max_words,
        max_sentence_length=args.max_sentence_length,
        cbow_file=cbow_path,
        sg_file=sg_path,
        normalization=args.normalization,
        out_dir=args.output_dir
    )

    # Run learning
    print("\nStarting learning process...")
    cxg.learn(
        input_data=args.grammar_corpus,
        npmi_threshold=args.npmi_threshold,
        starting_index=args.starting_index,
        min_count=args.min_count,
        max_vocab=args.max_vocab,
        cbow_range=args.cbow_range,
        sg_range=args.sg_range,
        get_examples=args.get_examples,
        increments=args.increments,
        learning_rounds=args.learning_rounds,
        forgetting_rounds=args.forgetting_rounds,
        cluster_only=args.cluster_only
    )

    print("\nLearning complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
