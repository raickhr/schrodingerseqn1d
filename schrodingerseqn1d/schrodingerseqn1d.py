# -*- coding: utf-8 -*-
import sys
import os
path = os.getcwd()
sys.path.append(path + '/schrodingerseqn1d')
import tensorflow as tf
#from schrodingerseqn1d 
import Hmatrix as H
#from schrodingerseqn1d 
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

dx = (endX - startX)/(numberOfNodes)
domain = tf.range(startX+dx/2, endX + dx/2, delta=dx, dtype=tf.float64)
#print('DOMAIN SHAPE = ',tf.shape(domain))
L = endX - startX
normalizedDomain = (domain - startX) * 2* math.pi /L

#print('Normalized Domain',normalizedDomain)
numberOfBasis = inp.basisSize
potential = tf.constant(inp.potential, dtype=tf.float64)


phi = H.get_phi(numberOfBasis, numberOfNodes, normalizedDomain)
H_mat = H.get_H_matrix(phi, c, dx, potential, numberOfNodes, numberOfBasis)

energy, Coefficents = H.get_lowestEnergy(H_mat)
sess = tf.Session()
with sess.as_default():
    import numpy as np
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    print(' \n \nH_matrix = ')
    print( H_mat.eval())
    print ('\n \n')
    waveFunction = tf.zeros([1,numberOfNodes],dtype = tf.float64)
    for i in range(0,numberOfBasis):
        phi_i = tf.gather(phi, tf.constant(i))
        waveFunction = waveFunction + Coefficents[i] * phi_i
    PlotVal = waveFunction.eval()[0,:]
    XVal = domain.eval()
    fig = plt.figure(figsize=(10,6))
    plt.plot(XVal,PlotVal)
    plt.xlabel('Domain')
    plt.ylabel('Wave Function')
    plt.title('Wave Function Corresponding to Lowest Energy ' + str(energy))
    plt.savefig(path + '/WaveFunction.png')
    #plt.show()
