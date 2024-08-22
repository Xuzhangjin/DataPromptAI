# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:15:42 2024

@author: zhang
"""

# setup.py


# setup.py
from setuptools import setup, find_packages

setup(
    name="DataPromptAI",
    version="0.1.0",
    author="Zhangjin Xu",
    author_email="zhangjinxu24@gmail.com",
    description="A Python package to generate and execute code snippets based on user prompts.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Xuzhangjin/DataPromptAI.git",  # Update this with your repo link
    packages=find_packages(),
    install_requires=[
        "openai",
        "pandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
