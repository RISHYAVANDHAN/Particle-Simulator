
from setuptools import setup, find_packages

setup(
    name="particle-simulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "taichi",
        "numpy",
        "matplotlib",
        "scipy"
        # Add other dependencies as needed
    ],
    description="Particle physics simulation package",
    python_requires=">=3.6",
)
