#!/usr/bin/env python3
"""
Script to download ProtBert model locally for Docker builds
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModel, AutoConfig


def main():
    parser = argparse.ArgumentParser(description="Download ProtBert model and tokenizer")
    parser.add_argument(
        "-o", "--output", 
        required=False, 
        default="./protbert/models--Rostlab--prot_bert_bfd/snapshots/main",
        help="Output directory to save the model (default: ./protbert/models--Rostlab--prot_bert_bfd/snapshots/main)"
    )
    parser.add_argument(
        "-m", "--model",
        required=False,
        default="Rostlab/prot_bert_bfd",
        help="Model name to download (default: Rostlab/prot_bert_bfd)"
    )
    
    args = parser.parse_args()
    
    # Create directory for the model
    model_dir = args.output
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Downloading {args.model} model and tokenizer...")
    print(f"Output directory: {os.path.abspath(model_dir)}")
    
    try:
        # Download and save the model
        print("Downloading model...")
        model = AutoModel.from_pretrained(args.model)
        model.save_pretrained(model_dir)
        print(f"✓ Model saved to {model_dir}")
        
        # Download and save the tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(model_dir)
        print(f"✓ Tokenizer saved to {model_dir}")
        
        # Download and save the config
        print("Downloading config...")
        config = AutoConfig.from_pretrained(args.model)
        config.save_pretrained(model_dir)
        print(f"✓ Config saved to {model_dir}")
        
        print(f"\nModel successfully downloaded to: {os.path.abspath(model_dir)}")
        print("\nFiles downloaded:")
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  {file}: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("Make sure you have transformers and torch installed:")
        print("pip install transformers torch")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())