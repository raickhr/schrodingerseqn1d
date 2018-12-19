#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', ]

setup_requirements = ['pytest-runner','tensorflow' ]

test_requirements = ['pytest', ]

setup(
    author="Shikhar Rai",
    author_email='shikhar.rai@rochester.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Solution of Schrodingers Equation in 1D. University of Rochester Assignment CHE447",
    entry_points={
        'console_scripts': [
            'schrodingerseqn1d=schrodingerseqn1d.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='schrodingerseqn1d',
    name='schrodingerseqn1d',
    packages=find_packages(include=['schrodingerseqn1d']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/raickhr/schrodingerseqn1d',
    version='0.1.0',
    zip_safe=False,
)