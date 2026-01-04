# tests/__init__.py
"""
Test Suite for Face Direction Classifier
=======================================

This package contains unit and integration tests for the
Face Direction Classifier application.
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test configuration
TEST_CONFIG = {
    "test_images_dir": "test_images",
    "model_path": "models/face_direction_classifier.pth",
    "test_timeout": 30,  # seconds
    "coverage_target": 80  # percentage
}

# Test data
TEST_IMAGES = [
    "test_front.jpg",
    "test_side.jpg",
    "test_back.jpg"
]

# Skip tests if dependencies not available
try:
    import torch
    import torchvision
    import streamlit
    import pytest
    TEST_DEPENDENCIES_AVAILABLE = True
except ImportError:
    TEST_DEPENDENCIES_AVAILABLE = False
    print("Warning: Some test dependencies not available. Tests may fail.")

# Import test modules
from .test_inference import TestFaceDirectionClassifier
from .test_app import TestStreamlitApp

__all__ = [
    "TestFaceDirectionClassifier",
    "TestStreamlitApp",
    "TEST_CONFIG",
    "TEST_IMAGES"
]

print("Test suite initialized for Face Direction Classifier.")