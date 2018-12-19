import numpy as np
import sys

class config:
    basisSize = 0
    numOfNodes = 0
    constant_c = 0
    domain = []
    potential = []

    def __init__(self, file):
        self.domain = []
        self.potential = []

        f = open(file, "r")
        for line in f.readlines():
            line_read = line.split('=')
            # if line_read[0] == '\n':
            #     continue
            left = line_read[0].lstrip().rstrip()
            right = line_read[1].lstrip().rstrip()

            if left.upper() == 'C':
                self.constant_c = int(right)

            if left.upper() == 'BASIS SIZE':
                self.basisSize = int(right)

            if left.upper() == 'NO OF NODES':
                self.numOfNodes = int(right)

            if left.upper() == 'DOMAIN':
                start,end = right.split(',')
                start = float(start.lstrip('['))
                end = float(end.rstrip(']'))
                self.domain = [start,end]

            if left.upper() == 'POTENTIAL':
                right = right.lstrip('[').rstrip(']')
                for i in right.split(','):
                    val = float(i.lstrip().rstrip())
                    self.potential.append(val)
                # if len(self.potential) != self.numOfNodes:
                #     #print(type(len(self.potential)))
                #     #print(type(self.numOfNodes))
                #     print('number of nodes = ',self.numOfNodes)
                #     print('potential size = ',len(self.potential))
                #     print('ERROR !! Number of nodes and size of potential array mismatch')
                    
                    #sys.exit()
                
            
        f.close()


#config_data = config("config.txt")
