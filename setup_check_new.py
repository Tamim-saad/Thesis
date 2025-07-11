"""
ANSYS Mesh Data ML Analysis - Setup Check
==========================================

This script checks if all required packages are installed and sets up the environment
for the ANSYS mesh data machine learning analysis.

Run this script before opening the Jupyter notebook to ensure everything is properly configured.

Usage:
    python setup_check.py
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name}")
        return False

def install_package(package_name):
    """Install a package using pip"""
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def check_gpu():
    """Check GPU availability"""
    print("🖥️ Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} ({gpu_count} device(s))")
            return True
        else:
            print("⚠️ No GPU available - will use CPU (training will be slower)")
            return False
    except ImportError:
        print("❓ Cannot check GPU - PyTorch not installed")
        return False

def check_data_directory():
    """Check if data directory exists"""
    print("📁 Checking data directory...")
    data_dir = Path("4_bonemat_cdb_files")
    if data_dir.exists():
        cdb_files = list(data_dir.glob("*.cdb"))
        print(f"✅ Data directory found with {len(cdb_files)} CDB files")
        return True
    else:
        print("⚠️ Data directory '4_bonemat_cdb_files' not found")
        print("Please ensure your CDB files are in the correct directory")
        return False

def main():
    """Main setup check function"""
    print("=" * 60)
    print("ANSYS Mesh Data ML Analysis - Setup Check")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        return False
    
    # Define required packages
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
        ("jupyter", "jupyter"),
        ("tqdm", "tqdm")
    ]
    
    print("\nChecking required packages...")
    missing_packages = []
    
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Found {len(missing_packages)} missing packages")
        install = input("Would you like to install them? (y/n): ")
        if install.lower() in ['y', 'yes']:
            for package in missing_packages:
                install_package(package)
        else:
            print("⚠️ Please install missing packages manually using:")
            print("pip install -r requirements.txt")
    
    # Check GPU
    check_gpu()
    
    # Check data directory
    check_data_directory()
    
    print("\n" + "=" * 60)
    print("Setup check complete!")
    print("\nNext steps:")
    print("1. Open the 'ANSYS_Mesh_ML_Analysis.ipynb' notebook")
    print("2. Run the cells step by step")
    print("3. Modify the CDB file path to point to your data")
    print("4. Experiment with different models and parameters")
    print("=" * 60)

if __name__ == "__main__":
    main()
