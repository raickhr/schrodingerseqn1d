#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `schrodingerseqn1d` package."""
import sys
import os
path = os.getcwd()
sys.path.append(path + '/schrodingerseqn1d')


import unittest
import pytest

from schrodingerseqn1d import Hmatrix
from schrodingerseqn1d import readConfig

from click.testing import CliRunner

from schrodingerseqn1d import schrodingerseqn1d
from schrodingerseqn1d import cli

import tensorflow as tf
import math


@pytest.fixture
def response():
    """Sample pytest fixture.
    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'schrodingerseqn1d.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output

class test_ckhr(tf.test.TestCase):
    def test_get_phi(self):
        #unittest for the function get_phi
        numberOfBasis = 5
        numberOfNodes = 6
        dx = 2*math.pi/(numberOfNodes)
        domain = tf.range(-math.pi+dx/2,math.pi+dx/2, delta=dx, dtype=tf.float64)
        phi = Hmatrix.get_phi(numberOfBasis, numberOfNodes, domain)
        
        sess = tf.Session()
        with sess.as_default():
            mockPhi = tf.constant(\
            [[    1,      1,      1,      1,      1,      1 ],\
            [-0.866, -0.000,  0.866,  0.866,  0.000, -0.866 ],\
            [  -0.5,   -1.0,   -0.5,    0.5,      1,    0.5 ],\
            [   0.5,  - 1.0,    0.5,    0.5,   -1.0,    0.5 ],\
            [ 0.866,  0.000, -0.866,  0.866,  0.000, -0.866 ]], dtype = tf.float64)

            mock = mockPhi
            test = phi.eval()
            difference =  test - mock
            err = tf.math.abs(tf.reduce_sum(difference))
            self.assertAllEqual(err < 0.01, True)
        
    def test_get_H_phi(self):
        #unittest for the function get_H_phi
        numberOfBasis = 5
        numberOfNodes = 6
        c = tf.constant(1, dtype= tf.float64)
        dx = 2*math.pi/(numberOfNodes)
        domain = tf.range(-math.pi+dx/2, math.pi+dx/2,\
        delta=dx, dtype=tf.float64)
        potential = tf.ones(numberOfNodes,dtype = tf.float64)
        
        phi = Hmatrix.get_phi(numberOfBasis, numberOfNodes, domain)
        H_phi = Hmatrix.get_H_phi(phi, c, potential, numberOfNodes, numberOfBasis)

        mockH_Phi = tf.constant(\
        [[ 1.000,  1.000,  1.000,  1.000,  1.000,  1.000], \
         [-1.732,  0.000,  1.730,  1.732,  0.000, -1.732],\
         [-1.000, -2.000, -1.000,  1.000,  2.000,  1.000],\
         [ 2.500, -5.000,  2.500,  2.500, -5.000,  2.500],\
         [ 4.330,  0.000, -4.330,  4.330,  0.000, -4.3301]],dtype= tf.float64)
        sess = tf.Session()
        with sess.as_default():
            test = H_phi.eval()
            mock = mockH_Phi.eval()
            difference = test - mock
            err = tf.math.abs(tf.reduce_sum(difference))    ## cummulative error
            self.assertAllEqual(err < 0.01, True)
            
    def test_get_H_matrix(self):
        #unittest for the function get_H_matrix
        numberOfBasis = 5
        numberOfNodes = 6
        c = tf.constant(1, dtype=tf.float64)
        dx = 2*math.pi/(numberOfNodes)
        domain = tf.range(-math.pi+dx/2, math.pi+dx/2,
                          delta=dx, dtype=tf.float64)
        potential = tf.ones(numberOfNodes, dtype=tf.float64)

        phi = Hmatrix.get_phi(numberOfBasis, numberOfNodes, domain)
        H_matrix = Hmatrix.get_H_matrix(\
            phi, c, dx, potential, numberOfNodes, numberOfBasis)

        mockH_matrix = tf.constant([\
        [6.283,  0.000,  0.000,  0.000,  0.000],\
        [0.000,  6.283,  0.000,  0.000,  0.000],\
        [0.000,  0.000,  6.283,  0.000,  0.000],\
        [0.000,  0.000,  0.000, 15.708,  0.000],\
        [0.000,  0.000,  0.000,  0.000, 15.708]]\
        , dtype=tf.float64)
        sess = tf.Session()
        with sess.as_default():
            test = H_matrix.eval()
            mock = mockH_matrix.eval()
            difference = test - mock
            err = tf.math.abs(tf.reduce_sum(difference)) ## cummulative error
            self.assertAllEqual(err < 0.01, True)

    def test_get_lowestEnergy(self):
        #This function test the lowest energy and the eigenvalues
        H_matrix = tf.constant(\
        [[1,    2,     3],\
        [2,     3,     4],\
        [3,     4,     5]], dtype= tf.float64)
        
        E,V = Hmatrix.get_lowestEnergy(H_matrix)
        mockE = tf.constant(-0.6235,dtype=tf.float64)
        mockV = tf.constant([0.8277, 0.1424, -0.5428],dtype = tf.float64)
        sess = tf.Session()
        with sess.as_default():
            test1 = E
            mock1 = mockE.eval()
            difference = test1 - mock1
            #print
            err = tf.math.abs(tf.reduce_sum(difference))
            self.assertAllEqual(err < 0.01, True)

            test2 = V
            mock2 = mockV.eval()
            difference = test1 - mock1
            #print
            err = tf.math.abs(tf.reduce_sum(difference))
            self.assertAllEqual(err < 0.01, True)

    def test_readConfig(self):
        # This function check the input taken from the file
        filename = path +'/mockConfig.txt'
        inp = readConfig.config(filename)
        basisSize = inp.basisSize
        numOfNodes = inp.numOfNodes
        constant_c = inp.constant_c
        domain = inp.domain
        potential = inp.potential

        mockC = 1
        mockBasisSize = 7
        mockNumNodes = 10
        mockDomain= [-1, 1]
        mockPotential = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.assertAllEqual(basisSize, mockBasisSize)
        self.assertAllEqual(constant_c, mockC)
        self.assertAllEqual(numOfNodes, mockNumNodes)
        self.assertAllEqual(domain, mockDomain)
        self.assertAllEqual(potential, mockPotential)


if __name__ == '__main__':
    unittest.main()
