
#####################################
#####################################
#### Blockchain AI Vehicle System ###
###### Created by Anthony Asilo #####
############ aka pillared ###########
########### December 2020 ###########
#### https://github.com/pillared ####
##### https://anthonyasilo.com ######
#####################################
#####################################

"""
When objects within 100m radius (close proximity), then detect whether they will collide or not. If yes, change route.
Square/box - ice hockey table - 7 pucks, fling and wing, can u predict which ones will hit eachother
"""


from math import acos,sin,cos,radians,degrees,atan2,asin,acos,pi,sqrt,isclose
import random
import numpy as np
import pandas as pd
import csv


#Global variables

radar_radius = .1    # radius of radar detection is 100 meters (.1 km) #.85 also works nice!
radar_update = .5     # time delay between each location update for vehicle 
r = 6371.137          # radius of earth at equator in kilometers
azimuth = 0           # global var for azimuth
data = pd.DataFrame()

#init columns to append data to to add to dataframe and then export as csv
host_latitude = []
host_longitude = []
host_speed = []
host_azimuth = []
host_distance = []
host_time = []

guest_latitude = []
guest_longitude = []
guest_speed = []
guest_azimuth = []
guest_distance = []
guest_time = []

intersection_latitude = []
intersection_longitude = []
collision = []


# AZIMUTH RULE --> NORTH = 0°, SOUTH = 180°, EAST = 90°, WEST = 270°   
class Vehicle:  
    #vehicle contstructor initializes based on name, hashid, x and y coords, and velocity (speed and direction)
    #initialize with speed 0, and motion false. Starting vehicles will allow them to move
    def __init__(self, name, ID, x, y, s, d):  
        self.id = ID  
        self.name = name
        self.coords = [x, y]
        self.velocity = [s, d]
        self.motion = True
        self.reward = None
    #function prints a description of the vehicle
    def get_velocity(self): 
        print("getter method called") 
        return self.velocity[0] 
       
     # function to set value of _age 
    def set_velocity(self, a): 
        print("setter method called") 
        self.velocity[0] = a 

    # function to delete _age attribute 
    def del_velocity(self): 
        del self.velocity[0] 

    speed = property(get_velocity, set_velocity, del_velocity)

    def get_direction(self): 
        print("getter method called") 
        return self.velocity[1] 
       
     # function to set value of _age 
    def set_direction(self, a): 
        print("setter method called") 
        self.velocity[1] = a 

    # function to delete _age attribute 
    def del_direction(self): 
        del self.velocity[1] 

    direction = property(get_direction, set_direction, del_direction)

    def describe(self):  
        print("Vehicle [%s] at (%f , %f) moving %s° at %d KPH \n" % (self.id, self.coords[0], self.coords[1], self.velocity[1], self.velocity[0]))  
    
#with two coordinated, fine the distance between them on the sphere and the azimuth (angle of first coord relative to north!)
def inverseCoords(lat1, lon1, lat2, lon2): 
      
    # The math module contains a function named 
    # radians which converts from degrees to radians. 
    lon1 = radians(lon1) 
    lon2 = radians(lon2) 
    lat1 = radians(lat1) 
    lat2 = radians(lat2) 
       
    # Haversine formula  
    dlon = lon2 - lon1  
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    global azimuth
    azimuth = degrees(atan2(sin(dlon) * cos(lat2),cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)))
    if(azimuth < 0):
        azimuth += 360
    elif(azimuth > 360):
        azimuth -= 360
    c = 2 * asin(sqrt(a))  
     
    # Radius of earth in kilometers. Use 3956 for miles 
    #r = 6371
    # calculate the result
    #print(str(azimuth) + "°")
    return (c * r , azimuth) 

# Given a start coordinate with a direction and distance, find the end coordinate!
def terminalCoords(lat1, lon1, azimuth, dis):
    #r = 6371 #Radius of the Earth
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    azimuth = radians(azimuth)
    
    lat2 = asin( sin(lat1) * cos(dis/r) + cos(lat1) * sin(dis/r) * cos(azimuth))
    lon2 = lon1 + atan2( sin(azimuth) * sin(dis/r) * cos(lat1), cos(dis/r) - sin(lat1) * sin(lat2))
    
    lat2 = degrees(lat2)
    if(lat2 < -180):
        lat2 += 360
    elif(lat2 > 180):
        lat2-= 360
    
    lon2 = degrees(lon2)
    if(lon2 < -180):
        lon2 += 360
    elif(lon2 > 180):
        lon2-= 360

    tCoord = ( lat2, lon2 )
    
    return tCoord

# Takes a Latitude and longitude and converts into a cartesian x y z
def cartesian(lat,lon):
    return [cos(lat)*cos(lon), sin(lat)*cos(lon), sin(lon)]

# Takes Cartesiian x y z in radians and returns a latlon in degrees
def LatLon(point):
    return (degrees(atan2(point[1],point[0])) , degrees(atan2(point[2], sqrt(point[0]**2 + point[1]**2))) )

# Whichever point if closer to the point 
def checkIfLies(pI, pF):
    pI = np.array(pI)
    pF = np.array(pF)
    theta_pIpF = acos(np.dot(pI, pF)/(sqrt(pI[0]**2 + pI[1]**2 + pI[2]**2) * sqrt(pF[0]**2 + pF[1]**2 + pF[2]**2)))
    return theta_pIpF

# Takes two arcs of great circles and finds the point of intersection
# Say that 1 and 2 are the start and end points for the first arc
# and that 3 and 4 and the start and end points for the second arc
def GCI(lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4):
    
    # Convert all points into radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    lat3 = radians(lat3)
    lon3 = radians(lon3)
    lat4 = radians(lat4)
    lon4 = radians(lon4)
    
    # Convert Lat Lon points into Cartesian X Y Z points 
    a0 = np.array(cartesian(lat1,lon1))
    a1 = np.array(cartesian(lat2,lon2))
    b0 = np.array(cartesian(lat3,lon3))
    b1 = np.array(cartesian(lat4,lon4))
    
    # Cross Product of arcs
    p = np.cross(a0,a1)
    q = np.cross(b0,b1)
    
    # Evaluating the bounds of the intersection
    h = (((q[0] * p[2] ) - (q[2] * p[0])) / ((q[1] * p[0]) - (q[0] * p[1])))
    g = ( ((-1) * (p[1] * h) - p[2] )/ p[0] )
    k = sqrt((r**2)/(g**2 + h**2 + 1))
    
    # Two possible intersections in the form of cartesian
    xo1 = [ g * k , h * k , k ]
    xo2 = [ -g * k , -h * k , -k ]
    #print(xo1)
    #print(xo2)
    
    # POI is Point Of Intersection
    POI = None     
    
    # booleans checking which of the two points on sphere is the actual intersection
    bool_axo1 = isclose(degrees(checkIfLies(a0,xo1)) + degrees(checkIfLies(a1,xo1)), degrees(checkIfLies(a0,a1)), abs_tol=1e-8)
    bool_bxo1 = isclose(degrees(checkIfLies(b0,xo1)) + degrees(checkIfLies(b1,xo1)), degrees(checkIfLies(b0,b1)), abs_tol=1e-8)
    bool_axo2 = isclose(degrees(checkIfLies(a0,xo2)) + degrees(checkIfLies(a1,xo2)), degrees(checkIfLies(a0,a1)), abs_tol=1e-8)
    bool_bxo2 = isclose(degrees(checkIfLies(b0,xo2)) + degrees(checkIfLies(b1,xo2)), degrees(checkIfLies(b0,b1)), abs_tol=1e-8)
    
    # boolean logic check, sets POI to point if and only of both points are valid along arc
    # POI is the cartesian converted back to a coordinate (Lat Lon)
    if( bool_axo1 and bool_bxo1 ):
        POI = LatLon(xo1)
    elif( bool_axo2 and bool_bxo2 ):
        POI = LatLon(xo2)
        
    return(POI)

"""
Function oppose gets the "opposing" direction on a coordinate plane 
    Function half gets the midangle between two angles (example is half(0,180) would be 90.)

    Using these together we can determine the approximate angle needed for a Guest vehicle
    to be positioned to possibly collide with the host vehicle when it is randomly 
    placed at a distance and angle away from the host vehicle
"""

#Method to calculate relative angle of random point near host vehicle to possible collision
def oppose(x):
    return x+180 if x < 180 else x-180

#Method to calculate relative angle of random point near host vehicle to possible collision
def half(x, y):
    
    if(abs(x-y) > 180):
        
        if(x >= y):
            print('x', x)
            x -= 360
    
        elif(y >= x):
            print('y', y)
            x += 360
    
    z = ((x + y) / 2)
    return z+360 if z < 0 else z

# Main driver method that runs all code.
def main():
    #define Host Vehicle
    HOST = Vehicle("Subaru Imprezza", "19i28b", 33.779398, -84.413279, 30, 270)
    HOST.describe()
    
    #define Guest Vehicle
    GUEST = Vehicle("Honda Accord", "1bofq9", 33.779322, -84.413278, 50, 90)
    GUEST.describe()
    
    # INVERSE COORDS - Retrieves distance and azimuth given two points
    lat1 = 33.779398 
    lat2 = 33.993333 
    lon1 = -84.413279 
    lon2 = -84.173888
    #r = 6371
    dis = inverseCoords(lat1, lon1, lat2, lon2)
    print("----------INVERSECOORD:----------")
    print("From (%f , %f) to (%f , %f)," % (lat1,lon1,lat2,lon2))
    print("The distance is %f K.M and the azimuth is %f°.\n" % (dis[0],dis[1])) 
    
    # TERMINAL COORDS - Retrieves end coord given start coord, azimuth, and distance
    brng = 42.823054 #Bearing is 90 degrees converted to radians.
    d = 32.469116 #Distance in km
    lat1 = 33.779398
    lon1 = -84.413279
    tcoord = terminalCoords(lat1,lon1,brng,d)
    print("----------TERMINALCOORD:----------")
    print("From (%f , %f)" % (lat1,lon1))
    print("with a distance of %f K.M and an azimuth of %f°," % (d, brng)) 
    print("the terminal coords is (%f , %f)\n" % (tcoord[0], tcoord[1]))
    
    #GREAT CIRCLE INTERSECTION - finds point of intersection given two arcs, each having a start and end coord
    lat1 = 33.779398 
    lon1 = -84.413279 
    lat2 = 33.993333 
    lon2 = -84.173888
    lat3 = 33.880452
    lon3 = -84.1588087
    lat4 = 33.890248
    lon4 = -84.4401807
    T = GCI(lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4)
    print("----------INTERSECTION:----------")
    print("With ArcPath1 with a startpoint of (%f , %f) and an endpoint of (%f , %f)" % (lat1,lon1,lat2,lon2))
    print("and ArcPath2 with a startpoint of (%f , %f) and an endpoint of (%f , %f)" % (lat3,lon3,lat4,lon4))
    print("The intersection is (%f , %f)\n" % (T[0],T[1]))
    
    # Based on HOST coordinates, generate GUEST vehicle a constant radius away from HOST but random angle,
    # then find approximate azimuth for GUEST needed to potentially collide into HOST vehicle.
    print("----------NEW:----------")
    HOST_A = Vehicle("Nissan Sentra", "914h80", 33.779398, -84.413279, 50, 0)
    HOST_A.describe()



    HOST_TIME_TO_INTERSECTION = 0
    GUEST_TIME_TO_INTERSECTION = 1
    BOOL_TIME = False
    BOOL_COUNT = 0
    while(BOOL_COUNT < 1000):
        
        #generate vehicle at a constant distance but at a random angle from the HOST vehicle, then use half() and oppose() to calculate possible angle for guest vehicle to drive in to collide with HOST
        dirawayfrom = random.randint(0,360)
        print(dirawayfrom)
        tcoord = terminalCoords(HOST_A.coords[0], HOST_A.coords[1], dirawayfrom, radar_radius)
        print(tcoord)
        GUEST_A = Vehicle("Nissan Altima", "rh0319", tcoord[0], tcoord[1], 50, half(HOST_A.velocity[1], oppose(dirawayfrom)))
        GUEST_A.describe()
        
        randy1 = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
        k = random.randint(0, 1)
        x = random.choices(randy1, weights=(10,10,10,10,10,10,10,10,10,10,10,5,5,5,2,2,2,1,1,1,1), k=1)
        if(k == 1):
            t = GUEST_A.velocity[0] + x[0]
            GUEST_A.set_velocity(t)
        else:
            t = GUEST_A.velocity[0] - x[0]
            GUEST_A.set_velocity(t)
        print("RANDY")
        print(GUEST_A.velocity[0])
        print(GUEST_A.velocity[1])
        if(GUEST_A.velocity[0] < 0 ):
            print("INRANDYINADAD")
            GUEST_A.set_velocity(abs(t))
            GUEST_A.set_direction(oppose(GUEST_A.velocity[1]))
        print(GUEST_A.velocity[0])
        print(GUEST_A.velocity[1])
        GUEST_A.describe()
        #find their intersection and then determine time for each one to get to intersection
        #distance half the earth arc length and then find intersection only for intersection, then scratch that and only care about points before intersection
        #distance of half earth
        #dhe = (180/360) * pi * r / 4
        dhe = 10
        print("DHE\t\t\t\t",dhe)
        print("HOST\t\t\t\t",HOST_A.coords[0], HOST_A.coords[1], HOST_A.velocity[1])
        print("GUEST\t\t\t\t",GUEST_A.coords[0], GUEST_A.coords[1], GUEST_A.velocity[1])
        HOST_A_TERMINAL = terminalCoords(HOST_A.coords[0], HOST_A.coords[1], HOST_A.velocity[1], dhe)
        print("HOSTTERM\t\t\t",HOST_A_TERMINAL,)
        GUEST_A_TERMINAL = terminalCoords(GUEST_A.coords[0], GUEST_A.coords[1], GUEST_A.velocity[1], dhe)
        print("GUESTTERM\t\t\t",GUEST_A_TERMINAL)        
        try:
            print("BEFORE T")
            T = GCI(HOST_A.coords[0],HOST_A.coords[1], HOST_A_TERMINAL[0],HOST_A_TERMINAL[1], GUEST_A.coords[0],GUEST_A.coords[1], GUEST_A_TERMINAL[0],GUEST_A_TERMINAL[1])
            
            #SET THEM NOT EQUAL SO LOOP INVARIANT IS TRUE
            print("IM INSIDE THE TRY")

            print("VELOCITY\t\t\t",GUEST_A.velocity[0])
            HOST_A_TO_POTENTIAL_COLLISION = inverseCoords(HOST_A.coords[0], HOST_A.coords[1], T[0], T[1])
            GUEST_A_TO_POTENTIAL_COLLISION = inverseCoords(GUEST_A.coords[0], GUEST_A.coords[1], T[0], T[1])
            print("HOSTPOTENTIAL\t\t\t",HOST_A_TO_POTENTIAL_COLLISION)
            print("GUESTPOTENTIAL\t\t\t",GUEST_A_TO_POTENTIAL_COLLISION)
            #TIME IN SECONDS FOR EACH VEHICLE TO GET TO INTERSECTION
            HOST_TIME_TO_INTERSECTION = (HOST_A_TO_POTENTIAL_COLLISION[0] / HOST_A.velocity[0]) * 60 * 60
            GUEST_TIME_TO_INTERSECTION = (GUEST_A_TO_POTENTIAL_COLLISION[0] / GUEST_A.velocity[0]) * 60 * 60
            print("HOST_TIME_TO_INTERSECTION\t", HOST_TIME_TO_INTERSECTION)
            print("GUEST_TIME_TO_INTERSECTION\t", GUEST_TIME_TO_INTERSECTION)
            print("BOOLTIME")
            BOOL_TIME = isclose(HOST_TIME_TO_INTERSECTION, GUEST_TIME_TO_INTERSECTION, abs_tol=.035)
            if(BOOL_TIME == True):
                BOOL_COUNT += 1
            print("BOOL_TIME\t\t\t", BOOL_TIME)
            print("BOOL_COUNT\t\t\t", BOOL_COUNT)
        except:
            print("COORDINATES NEVER INTERSECTED")
        print("T\t\t\t\t",T)
        print()
        host_latitude.append(HOST_A.coords[0])
        host_longitude.append(HOST_A.coords[1])
        host_speed.append(HOST_A.velocity[0])
        host_azimuth.append(HOST_A.velocity[1])
        host_distance.append(HOST_A_TO_POTENTIAL_COLLISION[0])
        host_time.append(HOST_TIME_TO_INTERSECTION)

        guest_latitude.append(GUEST_A.coords[0])
        guest_longitude.append(GUEST_A.coords[1])
        guest_speed.append(GUEST_A.velocity[0])
        guest_azimuth.append(GUEST_A.velocity[1])
        guest_distance.append(GUEST_A_TO_POTENTIAL_COLLISION[0])
        guest_time.append(GUEST_TIME_TO_INTERSECTION)
        if(T == None):
            intersection_latitude.append(None)
            intersection_longitude.append(None)
        else:
            intersection_latitude.append(T[0])
            intersection_longitude.append(T[1])
        collision.append(BOOL_TIME)
        print("---------------------------------------------------------------------")
    #if same time or within .25seconds, collision. mark action as True for collision, negative reward
    #if no collision, mark false for collision, positive reward.
    
    data["HOST_LAT"] = host_latitude
    data["HOST_LON"] = host_longitude
    data["HOST_AZI"] = host_azimuth
    data["HOST_SPEED"] = host_speed
    data["HOST_DIS"] = host_distance
    data["HOST_TIME"] = host_time

    data["GUEST_LAT"] = guest_latitude
    data["GUEST_LON"] = guest_longitude
    data["GUEST_AZI"] = guest_azimuth
    data["GUEST_SPEED"] = guest_speed
    data["GUEST_DIS"] = guest_distance
    data["GUEST_TIME"] = guest_time

    data["INTER_LAT"] = intersection_latitude
    data["INTER_LON"] = intersection_longitude
    data["COLLISION"] = collision
    print(data)
    data.to_csv('./data.csv') 
    
if __name__ == "__main__":
    main()
    
    