"""
FaithFed-XAI: Streamlit Clinical Decision Support & Research Dashboard.
"""
import sys, os
# Redirect to root app.py
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
from app import *
