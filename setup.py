# -*- coding: utf-8 -*-

from setuptools import setup

with open("README.md") as f:
    long_description = f.read()
with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    name="sparse",
    version="0.0.1",
    packages=['sparse', 'sparse.tests'],
    include_package_data=True,
    install_requires=install_requires,
    author="Danylo Ulianych",
    author_email="d.ulianych@gmail.com",
    description="Sparse representation solvers of P0-problem",
    long_description=long_description,
    license="MIT",
)
