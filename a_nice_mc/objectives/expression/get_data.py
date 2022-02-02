import numpy as np
from xy import *
import pickle, pprint
import matplotlib.pyplot as plt
import math
# from PIL import Image, ImageDraw, ImageFont

# def draw_grid(lattice_size=8,angle=[],beta=0):
#     height = 640
#     width = 640
#     image = Image.new(mode='L', size=(height, width), color=255)
#
#     # Draw some lines
#     draw = ImageDraw.Draw(image)
#     y_start = 0
#     y_end = image.height
#     step_size = int(image.width / lattice_size)
#
#     for x in range(0, image.width, step_size):
#         line = ((x, y_start), (x, y_end))
#         draw.line(line, fill=128)
#
#     x_start = 0
#     x_end = image.width
#
#     for y in range(0, image.height, step_size):
#         line = ((x_start, y), (x_end, y))
#         draw.line(line, fill=128)
#     # del draw
#     draw.text((10,10), "Temp %s" %(str(beta)), fill=(100))
#     a = step_size//2
#     pi = 3.141592654
#     for i in range(0, image.width, step_size):
#         for j in range(0, image.height, step_size):
#             draw.line(((i+a, j+a) , ( i + a + a*math.cos(2*pi*angle[j//step_size,i//step_size]), j + a - a*math.sin(2*pi*angle[j//step_size,i//step_size]))))
#             draw.line(((i+a, j+a) , ( i + a - a*math.cos(2*pi*angle[j//step_size,i//step_size]), j + a + a*math.sin(2*pi*angle[j//step_size,i//step_size]))))
#     image.show()

#Parameters might change
J = 1
max_t = 2.05
min_t = 0.05
lattice_shape = (8,8) #can be changed to (16,16) or (32,32)
steps = 1
iters_per_step = 15
random_state = 30
t_vals = np.linspace(min_t, max_t, 32)
print(t_vals) 
#n = 2
#t_vals = np.concatenate([t_vals[:13-n],t_vals[13+n:]])
# betas = 1 / T_vals
lattices = []
#Monte Carlo Simulation
for beta in t_vals:
        lat=[]
        print(beta)
        random_state=random_state+1
        xy=XYModelMetropolisSimulation(lattice_shape=lattice_shape,beta=1/beta,J=J,random_state=random_state)
        for q in range(13500):
            xy.simulate(steps,iters_per_step)
            lat.append(xy.L+0)
            # draw_grid(lattice_shape[0],xy.L,1/beta)
        lattices.append(lat[12000:])
        print('Done')
#Saving Data
output = open(str(lattice_shape)+'lattices.pkl', 'wb')
pickle.dump(lattices, output)
output.close()
