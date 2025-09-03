# FILE: test_imports.py
import sys
import numpy
import torch
import cv2
import albumentations
import skimage

print("--- Environment Verification ---")
print(f"Python version: {sys.version}")
print(f"NumPy version: {numpy.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Albumentations version: {albumentations.__version__}")
print(f"Scikit-image version: {skimage.__version__}")
print("-" * 30)

try:
    print("\nAttempting to import dataloaders.py and its dependencies...")
    # This sequence of imports mirrors the failure stack trace
    from dataloaders import create_dataloaders
    print("\n✅ SUCCESS: All critical modules imported without the NumPy conflict.")
    print("The environment is correctly configured.")

except ImportError as e:
    print(f"\n❌ FAILURE: Encountered an ImportError: {e}")
    print("This may indicate a missing OS-level dependency.")

except ValueError as e:
    print(f"\n❌ FAILURE: Encountered a ValueError: {e}")
    print("This is the classic NumPy binary incompatibility error.")
    print("If you see this, the requirements.txt pins did not work as expected.")

except Exception as e:
    print(f"\n❌ FAILURE: An unexpected error occurred: {e}")