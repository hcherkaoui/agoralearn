#! /usr/bin/env python
"""Setup module."""

# Authors: Hamza Cherkaoui

import os
import sys
from setuptools import setup, find_packages


def load_version():
    """Executes agoralearn/info.py in a globals dictionary and return it.
    Note: importing Agoralearn is not an option because there may be
    dependencies like prox_tv which are not installed and
    setup.py is supposed to install them.
    """
    globals_dict = {}
    with open(os.path.join("agoralearn", "info.py")) as fp:
        exec(fp.read(), globals_dict)
    return globals_dict


def is_installing():
    install_commands = set(["install", "develop"])
    return install_commands.intersection(set(sys.argv))


os.chdir(os.path.dirname(os.path.abspath(__file__)))

_VERSION_GLOBALS = load_version()
DISTNAME = "agoralearn"
DESCRIPTION = __doc__
LONG_DESCRIPTION = open("README.md").read()
MAINTAINER = "Hamza Cherkaoui"
MAINTAINER_EMAIL = "hamza.cherkaoui@telecom-sudparis.eu"
URL = "https://github.com/hcherkaoui/agoralearn"
LICENSE = "-"
VERSION = _VERSION_GLOBALS["__version__"]


if __name__ == "__main__":
    if is_installing():
        module_check_fn = _VERSION_GLOBALS["_check_module_dependencies"]
        module_check_fn(is_installing=True)

    install_requires = [
        "{0}>={1}".format(mod, meta["min_version"])
        for mod, meta in _VERSION_GLOBALS["REQUIRED_MODULE_METADATA"]
        if meta["required_at_installation"]
    ]

    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        zip_safe=False,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Programming Language :: Python :: 3.12",
        ],
        packages=find_packages(),
        install_requires=install_requires,
    )
