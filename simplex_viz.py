import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

def plot_fig_constraint():

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	height = 22000
	width = 200

	X = np.arange(0, width, 25)
	Y = np.arange(0, width, 25)
	X, Y = np.meshgrid(X, Y)

	def update_plot(plot_nnX, plot_nnY, plot_c1, plot_c2, plot_obj_func,plot_polytope):
	    
	    # Clear the plot before the update
	    ax.clear()
	    
	    ax.set_xlim(0, width)
	    ax.set_ylim(0, width)
	    ax.set_zlim(0, height)
	    
	    ax.set_xlabel('Number of products A')
	    ax.set_ylabel('Number of products B')
	    ax.set_zlabel('Profit (€)')
	    ax.view_init(45, -150)
	    
	    if plot_nnX:
	        # Plot the plane of non negativity constraint on X
	        nn_X_verts = [[0, width, 0],
	                [0, width, height],
	                [0, 0, height],
	                [0, 0, 0]]
	        nn_X = Poly3DCollection([nn_X_verts])
	        color = (1.,0.75,0.5,0.2)
	        nn_X.set_color(color)
	        ax.add_collection3d(nn_X)
	        
	    if plot_nnY:
	        # Plot the plane of non negativity constraint on Y
	        nn_Y_verts = [[width, 0, 0],
	                [width, 0, height],
	                [0, 0, height],
	                [0, 0, 0]]
	        nn_Y = Poly3DCollection([nn_Y_verts])
	        color = (1.,1.,0.,0.2)
	        nn_Y.set_color(color)
	        ax.add_collection3d(nn_Y)
	    
	    if plot_c1:
	        # Plot the constraint (1) 4X + 10Y = 800 #
	        c1_verts = [[0, 80, 0],
	                    [0, 80, height],
	                    [200, 0, height],
	                    [200, 0, 0]]
	        c1 = Poly3DCollection([c1_verts])
	        color = (0.,1.,0.,0.2)
	        c1.set_color(color)
	        ax.add_collection3d(c1)
	        
	    if plot_c2:
	        # Plot the constraint (2) 4X + 2Y = 320 #
	        c2_verts = [[0, 160, 0],
	                    [0, 160, height],
	                    [80, 0, height],
	                    [80, 0, 0]]
	        c2 = Poly3DCollection([c2_verts])
	        color = (1.,0.,1.,0.2)
	        c2.set_color(color)
	        ax.add_collection3d(c2)
	        
	    if plot_obj_func:
	        # Plot the plane of the objective function 50 X + 75 Y #
	        Z = 50 * X + 75 * Y
	        ax.plot_surface(X, Y, Z, alpha=0.5)
	        
	    if plot_polytope:
	        # Plot the polytope
	        points_coord_opt = np.array([[0,0,0],[0, 80,6000],[50, 60,7000],[80,0,4000],[0,0,0]])
	        points_coord_opt_low = np.array([[0,0,0],[0, 80,0],[50, 60,0],[80,0,0],[0,0,0]])

	        ax.plot(points_coord_opt[:,0], points_coord_opt[:,1], points_coord_opt[:,2], c='orange')
	        ax.plot(points_coord_opt_low[:,0], points_coord_opt_low[:,1], points_coord_opt_low[:,2], c='orange')
	        ax.plot([80, 80], [0, 0], [0,4000], c='orange')
	        ax.plot([50, 50], [60, 60], [0,7000], c='orange')
	        ax.plot([0, 0], [80, 80], [0,6000], c='orange')
	        

	interact(update_plot, plot_polytope=True, plot_nnX=False, plot_nnY=False, plot_c1=True, plot_c2=False, plot_obj_func=True);

def plot_fig_sens_analysis_obj_3D():
	fig_sa = plt.figure()
	ax = fig_sa.gca(projection='3d')

	unit_profit_A = widgets.IntSlider(min=0.0, max=140.0, step=1, value=50, description='Profit A (€/u): ')
	unit_profit_B = widgets.IntSlider(min=0.0, max=116.0, step=1, value=75, description='Profit B (€/u): ')

	def objective_function_eval(x,y):
	    return unit_profit_A.value*x+unit_profit_B.value*y

	def printer(unit_profit_A, unit_profit_B):
	    # Print the equation of the objective function    
	    #print("Objective function:")
	    #print
	    return
	    
	interact(printer,unit_profit_A=unit_profit_A, unit_profit_B=unit_profit_B);

	def update_x_range(*args):

	    # Clear the plot before the update
	    ax.clear()
	    
	    # Plot the polytope
	    points_coord_opt = np.array([[0,0,0],[0, 80,6000],[50, 60,7000],[80,0,4000],[0,0,0]])
	    points_coord_opt_low = np.array([[0,0,0],[0, 80,0],[50, 60,0],[80,0,0],[0,0,0]])

	    ax.plot(points_coord_opt[:,0], points_coord_opt[:,1], points_coord_opt[:,2], c='orange')
	    ax.plot(points_coord_opt_low[:,0], points_coord_opt_low[:,1], points_coord_opt_low[:,2], c='orange')
	    ax.plot([80, 80], [0, 0], [0,4000], c='orange')
	    ax.plot([50, 50], [60, 60], [0,7000], c='orange')
	    ax.plot([0, 0], [80, 80], [0,6000], c='orange')
	    
	    # Compute the profit of A regarding the profit of B while staying on the same solution.
	    unit_profit_A.value = (7000 - unit_profit_B.value * 60)/50
	    
	    points_coord = np.array([[0,0,0],[0, 80,0],[50, 60,0],[80,0,0]])
	    
	    # Intersection between X > 0 and Y > 0
	    points_coord[0] = [0, 0, 0]

	    # Intersection between constraint (1) and Y > 0, X = 0
	    points_coord[1] = [0, (c1_cap_max-4*0)/c1_y, 0]

	    # Intersection between contraint (1) and (2)
	    coeff_matrix = np.array([[c2_x,c2_y], [c1_x,c1_y]])
	    ordinate = np.array([c2_cap_max,c1_cap_max])
	    solution = np.linalg.solve(coeff_matrix, ordinate)
	    points_coord[2] = [solution[0], solution[1], 0]

	    # Intersection between constraint (2) and X > 0, Y = 0
	    points_coord[3] = [(c2_cap_max-2*0)/c2_x, 0, 0]

	    # Evaluate and plot the points
	    for p in points_coord:
	        p[2] = objective_function_eval(p[0],p[1])
	        ax.scatter(p[0],p[1], p[2])

	    # Plot the "plane" of the objective function
	    obj_func_plane = Poly3DCollection([np.ndarray.tolist(points_coord)])
	    color = (0.,0.,1.,0.2)
	    obj_func_plane.set_color(color) # BLUE #
	    ax.add_collection3d(obj_func_plane)

	    # Plot the constraint (1) 4X + 10Y = 800 #
	    c1_verts = [[points_coord[1,0], points_coord[1,1], 0],
	                points_coord[1],
	                points_coord[2],
	                [points_coord[2,0], points_coord[2,1], 0]]
	    c1 = Poly3DCollection([c1_verts])
	    color = (0.,1.,0.,0.2)
	    c1.set_color(color) # GREEN #
	    ax.add_collection3d(c1)

	    # Plot the constraint (2) 4X + 2Y = 320 #
	    c2_verts = [[points_coord[2,0], points_coord[2,1], 0],
	                points_coord[2],
	                points_coord[3],
	                [points_coord[3,0], points_coord[3,1], 0]]

	    c2 = Poly3DCollection([c2_verts])
	    color = (1.,0.,1.,0.2)
	    c2.set_color(color) # PURPLE #
	    ax.add_collection3d(c2)

	    ax.set_zlim(0, 10000)
	    ax.set_xlabel('Number of products A')
	    ax.set_ylabel('Number of products B')
	    ax.set_zlabel('Profit (€)')
	    
	    # Compute the optimal solution
	    x_A = points_coord[points_coord[:,2].argmax(0), 0]
	    x_B = points_coord[points_coord[:,2].argmax(0), 1]
	    s = points_coord[points_coord[:,2].argmax(0), 2]
	    ax.text3D(x_A, x_B, s, s="*", size="14")
	    
	    ax.text3D(points_coord[1][0],points_coord[1][1],points_coord[1][2],s="   "+ str(points_coord[1][2]) + ' €')
	    ax.text3D(points_coord[2][0],points_coord[2][1],points_coord[2][2],s="   "+ str(points_coord[2][2]) + ' €')
	    ax.text3D(points_coord[3][0],points_coord[3][1],points_coord[3][2],s="   "+ str(points_coord[3][2]) + ' €')
	    
	    # Print the constraint to show compliance
	    w1_hours = c1_x * x_A + c1_y * x_B
	    w2_hours = c2_x * x_A + c2_y *x_B
	    ax.text2D(0.6, 1, "Hours for workshop 1: " + str(w1_hours) + " h" , transform=ax.transAxes)
	    ax.text2D(0.6, 0.95, "Hours for workshop 2: " + str(w2_hours) + " h"  , transform=ax.transAxes)
	    
	    # Print the solution
	    ax.text2D(0., 0.95, "# of product A: " + str(x_A), transform=ax.transAxes)
	    ax.text2D(0., 0.90, "# of product B: " + str(x_B), transform=ax.transAxes)
	    ax.text2D(0., 0.85, "Profit: " + str(s) + " €", transform=ax.transAxes)

	    
	    # Print observation about sensitivity analysis
	    if points_coord[:,2].argmax(0) == 2:
	        ax.text2D(0., 1, "Original solution is still optimal !" , transform=ax.transAxes)
	    else:
	        ax.text2D(0., 1, "Original solution is no longer optimal !" , color="red", transform=ax.transAxes)
	    
	    #fig.canvas.draw(points_coord[2][0],points_coord[2][1],points_coord[1][0],points_coord[1][1])
	    return
	    
	unit_profit_B.observe(update_x_range, 'value')    

	# Constraint (1): 4X + 10Y = 800
	c1_x = 4
	c1_y = 10
	c1_cap_max = 800

	# Constraint (2) : 4X + 2Y = 320
	c2_x = 4
	c2_y = 2
	c2_cap_max = 320

	ax.set_zlim(0, 10000)
	ax.set_xlabel('Number of products A')
	ax.set_ylabel('Number of products B')
	ax.set_zlabel('Profit (€)')
	ax.view_init(45, -150)

	interact(update_x_range);

def plot_fig_sens_analysis_obj_2D():
	"""
	ERROR TO BE CORRECTED
	"""
	x_widget = widgets.FloatSlider(min=0.0, max=140.0, step=1, value=50)
	y_widget = widgets.FloatSlider(min=0.0, max=116.0, step=1, value=60)

	def printer(x, y):
	    return

	interact(printer,x=x_widget, y=y_widget);

	def update_x_range(*args):
	    x_widget.value = (7000 - y_widget.value * 50)/60
	    line.set_ydata((7000-y_widget.value*x)/x_widget.value)
	    fig.canvas.draw()
	y_widget.observe(update_x_range, 'value')    

	x = np.linspace(0, 200)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)

	line, = ax.plot(x, (60*x-7000)/50)

	# Plot the solution space
	polygon = Polygon([[0,0],[0, 80],[50, 60],[80,0]], facecolor=None)
	p = PatchCollection([polygon])
	p.set_alpha(0.3)
	ax.add_collection(p)

	# Plot the constraint (1) 4X + 10Y = 800 #
	plt.plot([0,200],[80,0], color='g', label='4X + 10Y = 800', linewidth=2)

	# Plot the constraint (2) 4X + 2Y = 320
	plt.plot([0,80],[160,0], color='m', label='4X + 2Y = 320', linewidth=2)

	ax.set_xlim(0, 150)
	ax.set_ylim(0, 150)

	interact(update_x_range);


