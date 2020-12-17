import os
from setuptools import setup

name = "my_scripts"

with open("README.md") as f:
    long_description = f.read()

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name=name,
    description="My stuff",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sosnowsky/my_scripts",
    author="Sosnowsky",
    author_email="juan.m.losada@uit.no",
    license="GPL",
    version="1.0",
    packages=["my_scripts"],
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    zip_safe=False,
)
