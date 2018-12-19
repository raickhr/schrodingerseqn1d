# -*- coding: utf-8 -*-
import sys
import os
path = os.getcwd()
sys.path.append(path + '/schrodingerseqn1d')
import tensorflow as tf
import Hmatrix as H
import readConfig
import matplotlib.pyplot as plt
import math

"""Main module."""

print('PATH = ',path)
filename = path +'/config.txt'
inp = readConfig.config(filename)

c = tf.constant(inp.constant_c,dtype=tf.float64)

numberOfNodes = inp.numOfNodes
startX = inp.domain[0]  
endX = inp.domain[1] 

dx = (endX - startX)/(numberOfNodes)        ### calculation the distance between any two pair of nodes
domain = tf.range(startX+dx/2, endX + dx/2, delta=dx, dtype=tf.float64)
#print('DOMAIN SHAPE = ',tf.shape(domain))
L = endX - startX          ### total domain lenght

### the domain is normalized
normalizedDomain = (domain - startX) * 2* math.pi /L   
#print('Normalized Domain',normalizedDomain)
numberOfBasis = inp.basisSize
potential = tf.constant(inp.potential, dtype=tf.float64)

## calculate basis functions
phi = H.get_phi(numberOfBasis, numberOfNodes, normalizedDomain) 

## Operate H on basis functions
H_mat = H.get_H_matrix(phi, c, dx, potential, numberOfNodes, numberOfBasis)  

### Solve eigenValue problem
energy, Coefficents = H.get_lowestEnergy(H_mat)  

sess = tf.Session()
with sess.as_default():
    ### USING numpy for output formatting
    import numpy as np
    np.set_printoptions(precision=2)  ## output formatting
    np.set_printoptions(suppress=True) ## supress scientific notation
    print(' \n \nH_matrix = ')
    print( H_mat.eval())   ## Print H matirx
    print ('\n \n')

    ## wavefunction superpostion of all basis function with its amplitude
    ## first eigenfunction is constant
    waveFunction = Coefficents[0] * tf.zeros([1, numberOfNodes], dtype=tf.float64)
    for i in range(1,numberOfBasis):
        phi_i = tf.gather(phi, tf.constant(i))
        waveFunction = waveFunction + Coefficents[i] * phi_i  ## superimpose each basis function with its amplitude
    PlotVal = waveFunction.eval()[0,:]
    XVal = domain.eval()
    fig = plt.figure(figsize=(10,6))
    plt.plot(XVal,PlotVal)
    plt.xlabel('Domain')
    plt.ylabel('Wave Function')
    plt.title('Wave Function Corresponding to Lowest Energy ' + str(energy))
    plt.savefig(path + '/WaveFunction.png')
    #plt.show()
