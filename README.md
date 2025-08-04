<p align="center">
    <a href="https://www.python.org/downloads/release/python-312/">
        <img alt="Python Version" src="https://img.shields.io/badge/python-3.12-blue.svg" style="margin-right: 50px;">
    </a>
    <a href="https://opensource.org/license/mit">
        <img alt="License: MIT" src="https://img.shields.io/badge/Licence-MIT-brightgreen" style="margin-right: 50px;">
    </a>
    <a href="https://dl.circleci.com/status-badge/redirect/circleci/GaE9Rv4PkJxh1MG3cLS17a/BBTHWFXBaYvJwcd4myeSTd/tree/main">
        <img alt="CircleCI Build Status" src="https://dl.circleci.com/status-badge/img/circleci/GaE9Rv4PkJxh1MG3cLS17a/BBTHWFXBaYvJwcd4myeSTd/tree/main.svg?style=svg" style="margin-right: 50px;">
    </a>
    <a href="https://codecov.io/gh/hcherkaoui/agoralearn">
        <img alt="Coverage" src="https://codecov.io/gh/hcherkaoui/agoralearn/graph/badge.svg?token=kTAoLIylSv" style="margin-right: 50px;">
    </a>
    <img alt="Last Commit" src="https://img.shields.io/github/last-commit/hcherkaoui/agoralearn" style="margin-right: 50px;">
</p>


# AgoraLearn

A collection of benchmarks for collaborative learning.

---

## Description:

**AgoraLearn** is a research-oriented Python benchmark suite focused on **collaborative learning**, in which models enhance their predictive performance by leveraging training data from other related datasets to improve generalization.

---

## 🔗 Important links

- Official source code repo: https://github.com/hcherkaoui/agoralearn
- MIT License: https://opensource.org/license/mit

---

##  🔧 Installation

In order install the package, run:
```bash
git clone https://github.com/hcherkaoui/agoralearn
cd agoralearn
pip install -r requirements.txt
pip install -e .
```

---

## 🧩 Dependencies

The required dependencies to use the software are:

 * numba (>=0.61.2)
 * numpy (>=2.2.6)
 * Pandas (>=2.3.1)
 * Torch (>=2.7.1)
 * Torchvision (>=0.22.1)
 * Transformers (>=4.53.1)
 * Joblib (>=0.16.0)
 * Scikit-learn (>=1.7.0)
 * Seaborn (>=0.13.2)
 * Matplotlib (>=3.10.0)

---

## 🚧 Development

In order to launch the unit-tests, run the command:
```bash
pytest  # run the unit-tests
```

In order to check the PEP 8 compliance level of the package, run the command::
```bash
flake8 --ignore=E501,W503 --count agoralearn
```

