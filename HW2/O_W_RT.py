import numpy as np
import matplotlib.pyplot as plt

women_x = np.array([[1.0,1928.0],[1.0,1932.0],[1.0,1936.0],[1.0,1948.0],[1.0,1952.0],
                    [1.0,1956.0],[1.0,1960.0],[1.0,1964.0],[1.0,1968.0],
                    [1.0,1972.0],[1.0,1976.0],[1.0,1980.0],[1.0,1984.0],
                    [1.0,1988.0],[1.0,1992.0],[1.0,1996.0],[1.0,2000.0],
                    [1.0,2004.0],[1.0,2008.0]])
women_y = np.array([[12.2],[11.9],[11.5],[11.9],[11.5],[11.5],[11.0],[11.4],
                    [11.0],[11.07],[11.08],[11.06],[10.97],[10.54],[10.82],
                    [10.94],[11.12],[10.93],[10.78]])


women_I = np.linalg.inv(np.dot(np.transpose(women_x),women_x))
women_a = np.dot(women_I,np.transpose(women_x))
women_b = np.dot(women_a,women_y)
print(women_b)
##women = np.dot(women_I,women_a)
##print(women)
