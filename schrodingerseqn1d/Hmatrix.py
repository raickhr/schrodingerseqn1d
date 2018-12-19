import tensorflow as tf
import math
#sess = tf.InteractiveSession()


def get_phi(numberOfBasis, numberOfNodes, domain):
    sess = tf.Session()
    with sess.as_default():
        numberOfBasis = numberOfBasis
        numberOfNodes = numberOfNodes
        # phi is basis function
        phi = tf.ones([1, numberOfNodes], dtype = tf.float64)  # First basis is a constant of one
        for i in range(1, numberOfBasis):
            angleCoeff = tf.constant((i-1)//2+1, dtype = tf.float64) # In basis cos(2x) 2 is the angle coeff
            args = angleCoeff * domain # argument of the sin and cosine function
            if i % 2 == 0:
                phi_update_row = tf.math.sin(args)   
            else:
                phi_update_row = tf.math.cos(args)
            phi = tf.concat([phi ,[phi_update_row]],0)  # add rows to the basis function
            #print(phi.eval())
        return phi


def get_H_phi(phi, c, potential, numberOfNodes, numberOfBasis):
    ## This function gives the output of H operated on basis funcitons.
    
    ## H is -c * d^2/dx^2 (.) + V(x)*(.) 

    ## This is the real H operator not the one Reiner gave us in the class
    sess = tf.Session()
    with sess.as_default():
        # H_phi is H operated on basis function
        H_phi = tf.reshape(potential,[1,numberOfNodes]) ## First row of 
        #print(H_phi.eval())
        for i in range(1, numberOfBasis):
            angleCoeff = tf.constant((i-1)//2+1, dtype = tf.float64)
            phi_update_row = tf.gather(phi, tf.constant(i))
            nablaSq_phi = -c * -angleCoeff**2 * phi_update_row  ## first part of H operated on phi
            H_update_row = nablaSq_phi + potential * phi_update_row ## second part of H operated on phi

            H_phi = tf.concat([H_phi, [H_update_row]], 0)
            #print(H_phi.eval())
        return H_phi

### H matrix

def get_H_matrix(phi, c, dx, potential, numberOfNodes, numberOfBasis):

    ## H matirx can be calculated as (Braket Notaion)
    
    #  H[i,j] = < phi[i,:] | H | phi[j,:] >    
    
    # phi[i,:] and phi[j,:] are row vectors

    # | H | phi[j,:] > is H operated on on phi[j,:] and is row vector

    # Finally  sum( dx * phi[i] * H | phi[j,:] >  ) give the value of H[i,j]. dx is distance between nodes
 
    sess = tf.Session()
    with sess.as_default():
        H_matrix = tf.zeros([numberOfBasis, numberOfBasis], dtype=tf.float64)
        #print(H_matrix.eval())
        H_phi = get_H_phi(phi, c, potential, numberOfNodes, numberOfBasis)
        for i_index in range(0, numberOfBasis):
            for j_index in range(0, numberOfBasis):
                H_phi_j = tf.gather(H_phi, tf.constant(j_index))  #  H | phi[j,:] >  {bracket notation}
                phi_i = tf.gather(phi, tf.constant(i_index))
                phi_i_H_phi_j = dx * phi_i * H_phi_j     ###< phi[i, :] | H | phi[j, :] >
                H_i_j = tf.reduce_sum(phi_i_H_phi_j)
                #print(H_i_j)
                indices = [[i_index, j_index]]
                shape = [numberOfBasis, numberOfBasis]
                dummy = tf.scatter_nd(indices, [H_i_j], shape)
                H_matrix = H_matrix + dummy
                #H_matrix = tf.assign(H_matrix, update)
        #print(H_matrix.eval())
        return H_matrix

def get_lowestEnergy(H_matrix):
    e,v = tf.linalg.eigh(H_matrix)   ## solve eigenvalue problem
    sess = tf.Session()
    with sess.as_default():
        EnergyLevels = e.eval()
        Coefficients = v.eval()

        return EnergyLevels[0], Coefficients[0, :]   ## the eigen values are already sorted in tf.linalg,eigh()

