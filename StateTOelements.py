import numpy
import numpy as np
from scipy.optimize import fsolve
import math
from sympy import *
def State_to_elements(Rx,Ry,Rz,Vx,Vy,Vz,time):
    r= np.array([Rx , Ry , Rz])
    v= np.array([Vx , Vy , Vz])
    Mod_r= np.sqrt(np.dot(r,r))
    Mod_v=np.sqrt(np.dot(v,v))
    radial_v=np.dot(r,v)/Mod_r
    #Note that if radial_v >0 sattelite is flying away from perigee
    specific_angular_momentum = np.cross(r,v)
    Mod_h= np.sqrt(np.dot(specific_angular_momentum,specific_angular_momentum))
    inclination= np.arccos(specific_angular_momentum[2]/Mod_h)
    Node_line_vector=np.cross(np.array([0,0,1]),specific_angular_momentum)
    Mod_Noadal_line=np.sqrt(np.dot(Node_line_vector,Node_line_vector))
    Capital_Omega = None
    if Node_line_vector[1] >= 0:
        Capital_Omega=np.arccos(Node_line_vector[0]/Mod_Noadal_line)

    elif Node_line_vector[1] < 0:
        Capital_Omega= 2*np.pi - np.arccos(Node_line_vector[0]/Mod_Noadal_line)
    Mu = 398600
    eccentricity_vector = np.add((1/Mu)*(np.cross(v,specific_angular_momentum)),-(r)/Mod_r)
    eccentricity = np.sqrt(np.dot(eccentricity_vector,eccentricity_vector))
    argument_of_perigee = None
    if eccentricity_vector[2] >= 0:
        argument_of_perigee= np.arccos(np.dot(Node_line_vector,eccentricity_vector)/(Mod_Noadal_line*eccentricity))
    elif eccentricity_vector[2] <0:
        argument_of_perigee = 2*np.pi - np.arccos(np.dot(Node_line_vector,eccentricity_vector)/(Mod_Noadal_line*eccentricity))
    real_anomaly =None
    if radial_v >=  0:
        real_anomaly = np.arccos(np.dot(eccentricity_vector,r)/(eccentricity*Mod_r))
    elif radial_v <0:
        real_anomaly=2*np.pi - np.arccos(np.dot(eccentricity_vector,r)/(eccentricity*Mod_r))
    semi_major = Mod_h*Mod_h/(Mu*(1-(eccentricity)**2))
    J2 = 1.08263*0.001
    Radius = 6378
    Var1 = 1-((eccentricity)**2)
    Numerator = -3*np.sqrt(Mu)*J2*Radius**2
    Denominator = 2*(semi_major**(3.5))*(Var1**(2))
    Omega_dot = (Numerator/Denominator)*np.cos(inclination)
    Arg_Perigee_Dot = (Omega_dot*(2.5*(np.sin(inclination))**(2) - 2))/(np.cos(inclination))
    Time_period = 2*np.pi*semi_major**(1.5)/np.sqrt(Mu)
    mean_motion = np.pi*2/Time_period
    x = symbols('x')
    Initial_Eccentric_Anamoly = solve(Eq(tan(x/2),(((1-eccentricity)/(1+eccentricity))**0.5)*tan(real_anomaly/2)),x)
    y = symbols('y')
    t_0 = solve(Eq(mean_motion*y,Initial_Eccentric_Anamoly[0]-eccentricity*sin(Initial_Eccentric_Anamoly[0])),y)
    t_f = t_0[0] + time
    final_time = math.modf(t_f/Time_period)[0]*Time_period
    Mean_Anamoly = mean_motion*final_time
    def func(a1):
        a2 = a1 -eccentricity*np.sin(a1) -Mean_Anamoly
        return a2
    final_eccentric_anamoly = fsolve(func,0.3)[0]
    a3 = symbols('a3')
    final_true_anamoly = solve(Eq(tan(a3/2),(((1+eccentricity)/(1-eccentricity))**0.5)*tan(final_eccentric_anamoly/2)),a3)
    new_r = np.array([cos(final_true_anamoly[0]),sin(final_true_anamoly[0]),0])
    new_r1 = np.multiply(new_r,Mod_h**2/(Mu*(1+eccentricity*cos(final_true_anamoly[0]))))
    new_v = np.array([-sin(final_true_anamoly[0]),eccentricity+cos(final_true_anamoly[0]),0])
    new_v1 = np.multiply(new_v,Mu/Mod_h)
    Capital_Omega_final = Capital_Omega + Omega_dot*time
    argument_of_perigee_final = argument_of_perigee  + Arg_Perigee_Dot*time
    First_Euler_matrix = np.array([[cos(argument_of_perigee_final),sin(argument_of_perigee_final),0],
                                   [-sin(argument_of_perigee_final),cos(argument_of_perigee_final),0]
                                  ,[0,0,1]])
    Second_Euler_matrix = np.array([[1,0,0],
                                    [0,cos(inclination),sin(inclination)],
                                    [0,-sin(inclination),cos(inclination)]])
    Third_Euler_matrix = np.array([[cos(Capital_Omega_final),sin(Capital_Omega_final),0],
                                   [-sin(Capital_Omega_final),cos(Capital_Omega_final),0],
                                   [0,0,1]])
    Rotation_matrix = np.dot(First_Euler_matrix,Second_Euler_matrix)
    r_matrix = np.dot(Rotation_matrix,Third_Euler_matrix)
    result = [[0,0,0],
              [0,0,0],
              [0,0,0]]
    for i in range(len(r_matrix)):
        for j in range(len(r_matrix[0])):
            result[j][i] = r_matrix[i][j]
    propagated_position = np.dot(result,new_r1)
    propagated_velocity = np.dot(result,new_v1)
    return [propagated_position,propagated_velocity]
print(State_to_elements(-3670,-3870,4400,4.700,-7.400,1.000,345600)[1])