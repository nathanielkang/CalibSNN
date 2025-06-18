#!/usr/bin/env python
"""
Simple script to download Adult and CoverType datasets for CalibSNN.
"""

import os
import sys
from prepare_missing_datasets import ensure_all_datasets_available

def main():
    """Download Adult and CoverType datasets."""
    print("="*60)
    print("CalibSNN Dataset Downloader")
    print("="*60)
    
    # List of datasets to download
    datasets = ['adult', 'covertype']
    
    print(f"\nPreparing to download: {', '.join(datasets)}")
    print("This may take a few minutes...\n")
    
    # Ensure all datasets are available
    success = ensure_all_datasets_available(datasets)
    
    if success:
        print("\n✅ All datasets downloaded successfully!")
        print("\nYou can now run CalibSNN experiments with:")
        print("  python run_CalibSNN.py --datasets adult --betas 0.05 0.1 0.3")
        print("  python run_CalibSNN.py --datasets covertype --betas 0.05 0.1 0.3")
    else:
        print("\n❌ Some datasets failed to download.")
        print("Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 