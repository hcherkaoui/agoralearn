"""Setup module."""

# Authors: Hamza Cherkaoui

from setuptools import setup, find_packages

setup(
    name='agoralearn',
    version='0.1.0',
    description='Collaborative time series forecasting library',
    author='Hamza Cherkaoui',
    author_email='hamza.cherkao@gmail.com',
    url='https://github.com/hcherkaoui/agoralearn',
    packages=find_packages(),
    python_requires='>=3.12.3',
    install_requires=['numpy',],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
)
