=================
SchrodingersEqn1D
=================


.. image:: https://img.shields.io/pypi/v/schrodingerseqn1d.svg
        :target: https://pypi.python.org/pypi/schrodingerseqn1d

.. image:: https://img.shields.io/travis/raickhr/schrodingerseqn1d.svg
        :target: https://travis-ci.org/raickhr/schrodingerseqn1d

.. image:: https://readthedocs.org/projects/schrodingerseqn1d/badge/?version=latest
        :target: https://schrodingerseqn1d.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/raickhr/schrodingerseqn1d/shield.svg
     :target: https://pyup.io/repos/github/raickhr/schrodingerseqn1d/
     :alt: Updates

.. image:: https://coveralls.io/repos/github/raickhr/schrodingerseqn1d/badge.svg?branch=master
:target: https://coveralls.io/github/raickhr/schrodingerseqn1d?branch=master



Solution of Schrodingers Equation in 1D. University of Rochester Assignment CHE447


* Free software: MIT license

Description
-----------
This program solves 1D Schrodinger's Equation and plots the WaveFunction for lowest Energy Level

Inputs
-----------
| Input is to provided in the active directory with text file config.txt in the working directory
| The value of constant c =  h/(2m) is to be given
| The number of fourier basis is to be provided. The fourier basis stars as 1, cos(x), sin(x), cos(2x), sin(2x) ...
| The number of nodes is to given
| Domain is to be input as list [start_location, end_locaiton]
| The potential values at each node is to be given

Sample config.txt
-----------------
| C = 1
| BASIS SIZE = 7
| NO OF NODES = 10
| DOMAIN = [-1,1]
| POTENTIAL = [1,2,3,4,5,6,7,8,9,10]

Command for use
--------------
At active directory 

python schrodingerseqn1d/schrodingerseqn1d.py

Output 
------
The output file is generated as WaveFunction.png. The Hamiltonian Matrix is printed in the console.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
