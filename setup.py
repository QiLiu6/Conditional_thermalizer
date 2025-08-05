#!/usr/bin/env python

from setuptools import setup, find_packages

description = "Conditional_thermalizer"
version="0.0.1"

setup(name="Conditional_thermalizer",
    version=version,
    description=description,
    url="https://github.com/QiLiu6/Conditional_thermalizer",
    author="Qi Liu",
    author_email="ql2221@nyu.edu",
    packages=find_packages(include=['CT']),
    )
