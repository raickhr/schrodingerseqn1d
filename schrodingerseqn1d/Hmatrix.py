import tensorflow as tf
import math
#sess = tf.InteractiveSession()


def get_phi(numberOfBasis, numberOfNodes, domain):
    sess = tf.Session()
    with sess.as_default():
        numberOfBasis = numberOfBasis
        numberOfNodes = numberOfNodes
        phi = tf.ones([1, numberOfNodes], dtype = tf.float64)  # phi is basis function
        for i in range(1, numberOfBasis):
            angleCoeff = tf.constant((i-1)//2+1, dtype = tf.float64)
            args = angleCoeff * domain 
            if i % 2 == 0:
                phi_update_row = tf.math.sin(args)
            else:
                phi_update_row = tf.math.cos(args)
            phi = tf.concat([phi ,[phi_update_row]],0)
            #print(phi.eval())
        return phi


def get_H_phi(phi, c, potential, numberOfNodes, numberOfBasis):
    sess = tf.Session()
    with sess.as_default():
        # H_phi is H operated on basis function
        H_phi = tf.reshape(potential,[1,numberOfNodes])
        #print(H_phi.eval())
        for i in range(1, numberOfBasis):
            angleCoeff = tf.constant((i-1)//2+1, dtype = tf.float64)
            phi_update_row = tf.gather(phi, tf.constant(i))
            nablaSq_phi = -c * -angleCoeff**2 * phi_update_row
            H_update_row = nablaSq_phi + potential * phi_update_row

            H_phi = tf.concat([H_phi, [H_update_row]], 0)
            #print(H_phi.eval())
        return H_phi

### H matrix

def get_H_matrix(phi, c, dx, potential, numberOfNodes, numberOfBasis):
    sess = tf.Session()
    with sess.as_default():
        H_matrix = tf.zeros([numberOfBasis, numberOfBasis], dtype=tf.float64)
        #print(H_matrix.eval())
        H_phi = get_H_phi(phi, c, potential, numberOfNodes, numberOfBasis)
        for i_index in range(0, numberOfBasis):
            for j_index in range(0, numberOfBasis):
                H_phi_j = tf.gather(H_phi, tf.constant(j_index))
                phi_i = tf.gather(phi, tf.constant(i_index))
                phi_i_H_phi_j = dx * phi_i * H_phi_j
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
    e,v = tf.linalg.eigh(H_matrix)
    sess = tf.Session()
    with sess.as_default():
        EnergyLevels = e.eval()
        Coefficients = v.eval()

        return EnergyLevels[0], Coefficients[0, :]






# def get_H_matrix2(phi, c, dx, potential, numberOfNodes, numberOfBasis):
#     sess = tf.Session()
#     with sess.as_default():
#         rows = 


#phi = get_phi(numberOfBasis, numberOfNodes, domain)
#H_matrix = get_H_matrix(phi, c, potential, numberOfBasis)
#print(H_matrix)
