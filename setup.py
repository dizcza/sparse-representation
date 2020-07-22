# -*- coding: utf-8 -*-

from setuptools import setup

with open("README.md") as f:
    long_description = f.read()
with open('requirements.txt') as f:
    install_requires = f.read()
extras_require = {}
with open('requirements-extra.txt') as f:
    extras_require['extra'] = f.read()


setup(
    name="sparse",
    version="0.0.1",
    packages=['sparse', 'sparse.tests'],
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    author="Danylo Ulianych",
    author_email="d.ulianych@gmail.com",
    description="Sparse representation solvers for P0- and P1-problems",
    long_description=long_description,
    license="MIT",
)
