from math import *
from controller import Robot
import cv2 as cv
import numpy as np
import json
import struct
from PIL import Image

class ErebusRobot:
    def __init__(self) -> None:
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.angle = 0
        self.max_velocity = 6.28
        self.tile = 6
        self.room4 = False
        self.map_matrix = [['5', '0', '5'],
                           ['0', '0', '0'],
                           ['5', '0', '5']]
        self.map_pos = [1, 1]
        self.passed = 0
        self.sent = False
        self.lack = False

        self.wall_positions = []
        self.not_visited = [] #not visited positions connected to a graph
        self.graph = {} #known connected positions on the map - {1: [2, 3]}

        self.victim_locations = []
        self.thresh = None
        self.ROI = None
        self.im_bw = None

        self.colourSensor = self.robot.getDevice('colour_sensor')
        self.wheel_left = self.robot.getDevice('wheel1 motor')
        self.wheel_right = self.robot.getDevice('wheel2 motor')
        self.ds_front_left = self.robot.getDevice('ds1')
        self.ds_front_centre_left = self.robot.getDevice('ds2')
        self.ds_front_centre_right = self.robot.getDevice('ds3')
        self.ds_front_right = self.robot.getDevice('ds4')
        self.ds_right_front = self.robot.getDevice('ds5')
        self.ds_right_back = self.robot.getDevice('ds6')
        self.ds_left_front = self.robot.getDevice('ds7')
        self.ds_left_back = self.robot.getDevice('ds8')
        self.camera_left = self.robot.getDevice('camera1')
        self.camera_right = self.robot.getDevice('camera2')
        self.camera_centre = self.robot.getDevice('camera3')
        self.gyro = self.robot.getDevice('gyro')
        self.gps = self.robot.getDevice('gps')
        self.receiver = self.robot.getDevice('receiver')
        self.emitter = self.robot.getDevice('emitter')
        self.left_encoder = self.wheel_left.getPositionSensor()
        self.right_encoder = self.wheel_right.getPositionSensor()


        self.receiver.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.gps.enable(self.timestep)
        self.camera_left.enable(self.timestep)
        self.camera_right.enable(self.timestep)
        self.camera_centre.enable(self.timestep)
        self.colourSensor.enable(self.timestep)
        self.ds_front_left.enable(self.timestep)
        self.ds_front_centre_left.enable(self.timestep)
        self.ds_front_centre_right.enable(self.timestep)
        self.ds_front_right.enable(self.timestep)
        self.ds_right_front.enable(self.timestep)
        self.ds_right_back.enable(self.timestep)
        self.ds_left_front.enable(self.timestep)
        self.ds_left_back.enable(self.timestep)
        self.left_encoder.enable(self.timestep)
        self.right_encoder.enable(self.timestep)
        self.wheel_left.setPosition(float("inf"))       
        self.wheel_right.setPosition(float("inf"))

        self.axes = {0: (0, 1), 45: (1, 1), 90: (1, 0), 135: (1, -1), 180: (0, -1), 225: (-1, -1), 270: (-1, 0), 315: (-1, 1)}
        self.map_axes = {0: {'l': (-2, -2), 'cl': (-1, -2), 'c': (0, -2), 'cr': (1, -2), 'r': (2, -2), 'bl': (-1, -1), 'br': (1, -1)}, 90: {'l': (-2, 2), 'cl': (-2, 1), 'c': (-2, 0), 'cr': (-2, -1), 'r': (-2, -2), 'bl': (-1, 1), 'br': (-1, -1)}, 180: {'l': (2, 2), 'cl': (1, 2), 'c': (0, 2), 'cr': (-1, 2), 'r': (-2, 2), 'bl': (1, 1), 'br': (-1, 1)}, 270: {'l': (2, -2), 'cl': (2, -1), 'c': (2, 0), 'cr': (2, 1), 'r': (2, 2), 'bl': (1, -1), 'br': (1, 1)}}
        self.camera_pos_ds = {'front': [self.ds_front_centre_left, self.ds_front_right, self.ds_front_left, self.ds_front_centre_right], 'left': [self.ds_left_front, self.ds_left_back], 'right': [self.ds_right_front, self.ds_right_back]}

        while self.robot.step(self.timestep) != -1:
            self.visited = [self.get_gps_coords(0)]
            self.graph[(lambda x: (x[0], x[1]))(self.get_gps_coords(0))] = []
            self.rests = (lambda x: [x[0] % 12, x[1] % 12])(self.get_gps_coords(0))
            self.set_default_angle()
            self.map_gps = (lambda x: [round(x[0] / self.tile, 0) * self.tile, round(x[1] / self.tile, 0) * self.tile])(self.get_gps_coords(3))
            self.mapping()
            break
    
    
    def get_surface(self):
        image = self.colourSensor.getImage()
        r = self.colourSensor.imageGetRed(image, 1, 0, 0)
        g = self.colourSensor.imageGetGreen(image, 1, 0, 0)
        b = self.colourSensor.imageGetBlue(image, 1, 0, 0)

        if r in range(225,254) and g in range(225,254) and b in range(225,254):
            return 0 #white
        elif r in range(0,60) and g in range(0,60) and b in range(0,60):
            return 2 #hole
        elif r in range(205,250) and g in range(170,220) and b in range(100,140):
            return 3 #swamp
        if r in range(254,256) and g in range(254,256) and b in range(254,256):
            return 4 #checkpoint
        elif r in range(60,90) and g in range(60,90) and b in range(245,256):
            return 6 #blue
        elif r in range(140,200) and g in range(40,90) and b in range(220,256):
            return 7 #purple
        elif r in range(20,50) and g in range(245,256) and b in range(20,50):
            return 8 #green
        elif r in range(245,256) and g in range(60,90) and b in range(60,90):
            return 9 #red
        
    def set_default_angle(self):
        last_left_rotation = self.left_encoder.getValue()
        last_gps = self.get_gps_coords(2)

        while self.robot.step(self.timestep) != -1:
            current_left_rotation = self.left_encoder.getValue() - last_left_rotation

            if current_left_rotation > 2:
                self.wheel_left.setVelocity(-self.max_velocity / 2)
                self.wheel_right.setVelocity(-self.max_velocity / 2)
                break
            
            self.wheel_left.setVelocity(self.max_velocity / 2)
            self.wheel_right.setVelocity(self.max_velocity / 2)

        gps = self.get_gps_coords(2)
        self.angle = (0 if last_gps[1] - gps[1] > 1 else 180 if last_gps[1] - gps[1] < -1 else 90 if last_gps[0] - gps[0] > 1 else 270) / 180 * pi
        
        while self.robot.step(self.timestep) != -1:
            current_left_rotation = self.left_encoder.getValue() - last_left_rotation

            if current_left_rotation < 0:
                self.wheel_left.setVelocity(0)
                self.wheel_right.setVelocity(0)
                break


    def movement(self, tiles):
        heading = self.get_heading()
        dir = self.axes[heading]
        start_gps = (lambda x: (round(x[0] / self.tile, 0) * self.tile, (round(x[1] / self.tile, 0) * self.tile)))(self.get_gps_coords(3))
        last_gps = (lambda x: x[0] if dir[0] in [1, -1] else x[1])(self.get_gps_coords(2))
        target_gps = (lambda x: (round((x[0] - dir[0] * 2 * tiles * self.tile) / self.tile, 0) * self.tile) if dir[0] in [1, -1] else (round((x[1] - dir[1] * 2 * tiles * self.tile) / self.tile, 0) * self.tile))(self.get_gps_coords(2))
        left_wall = False
        right_wall = False

        while self.robot.step(self.timestep) != -1:
            gps = (lambda x: x[0] if dir[0] in [1, -1] else x[1])(self.get_gps_coords(1))
            gyro = self.get_gyro_angle()
            if ((gps >= target_gps and tiles * -(dir[0] if dir[0] in [1, -1] else dir[1]) > 0) or (gps <= target_gps and tiles * -(dir[0] if dir[0] in [1, -1] else dir[1]) < 0) or (self.get_surface() in ([2, 8, 9] if not self.room4 else [2]) and tiles > 0)) or self.is_lack_of_progress():  
                self.passed = abs(gps - last_gps) / (self.tile * 2)
                break
            if 1 <= abs(gps - target_gps) <= 3 and self.ds_right_front.getValue() <= 0.12 and self.ds_right_back.getValue() >= 0.12 and not right_wall:
                right_wall = True
                self.wall_positions.append((start_gps[0] - self.axes[(heading + 315) % 360][0] * self.tile, start_gps[1] - self.axes[(heading + 315) % 360][1] * self.tile))
                self.visited.append([start_gps[0] - self.axes[(heading + 315) % 360][0] * self.tile, start_gps[1] - self.axes[(heading + 315) % 360][1] * self.tile])
            elif 1 <= abs(gps - target_gps) <= 3 and self.ds_left_front.getValue() <= 0.12 and self.ds_left_back.getValue() >= 0.12 and not left_wall:
                left_wall = True
                self.wall_positions.append((start_gps[0] - self.axes[(heading + 45) % 360][0] * self.tile, start_gps[1] - self.axes[(heading + 45) % 360][1] * self.tile))
                self.visited.append([start_gps[0] - self.axes[(heading + 45) % 360][0] * self.tile, start_gps[1] - self.axes[(heading + 45) % 360][1] * self.tile])

            if self.get_remaining_time() <= 3 and not self.sent:
                self.send_map()
                self.sent = True
            
            self.detect_victim(self.camera_left, 'left')
            self.detect_victim(self.camera_right, 'right')
            self.detect_victim(self.camera_centre, 'front')

            self.wheel_left.setVelocity(self.max_velocity * (-1 if tiles < 0 else 1))
            self.wheel_right.setVelocity(self.max_velocity * (-1 if tiles < 0 else 1))

        if not (self.ds_front_left.getValue() > 0.12 and self.ds_front_centre_left.getValue() > 0.12  and self.ds_front_centre_right.getValue() > 0.12 and self.ds_front_right.getValue() > 0.12 and self.is_not_visited(0)) or self.is_hole(self.camera_centre, 'centre'):
            self.wheel_left.setVelocity(0)
            self.wheel_right.setVelocity(0)
    

    def rotation(self, angle):
        last_gyro = round(self.get_gyro_angle()/45, 0) * 45

        while self.robot.step(self.timestep) != -1 and angle != 0:
            gyro = self.get_gyro_angle()
            if (angle > 0 and gyro > last_gyro + angle) or (angle < 0 and gyro < last_gyro + angle) or self.is_lack_of_progress():
                break
            if self.get_remaining_time() <= 3 and not self.sent:
                self.send_map()
                self.sent = True
            
            self.detect_victim(self.camera_left, 'left')
            self.detect_victim(self.camera_right, 'right')
            self.detect_victim(self.camera_centre, 'front')

            self.wheel_left.setVelocity((-self.max_velocity/2 if angle < 0 else self.max_velocity/2) * (1 if abs(angle) - abs(gyro - last_gyro) > 15 else 0.1))
            self.wheel_right.setVelocity((self.max_velocity/2 if angle < 0 else -self.max_velocity/2) * (1 if abs(angle) - abs(gyro - last_gyro) > 15 else 0.1))

        self.wheel_left.setVelocity(0)
        self.wheel_right.setVelocity(0)

    def graph_update(self):
        gps = (lambda x: (round(x[0] / self.tile, 0) * self.tile, round(x[1] / self.tile, 0) * self.tile))(self.get_gps_coords(3))
        heading = self.get_heading()

        if self.ds_front_left.getValue() > 0.12 and self.ds_front_centre_left.getValue() > 0.12  and self.ds_front_centre_right.getValue() > 0.12 and self.ds_front_right.getValue() > 0.12 and self.is_not_visited(0) and not self.is_hole(self.camera_centre, 'centre'):
            target = (gps[0] - self.tile * self.axes[heading][0], gps[1] - self.tile * self.axes[heading][1])
            self.graph[gps].append(target)
            self.graph[target] = [gps] if target not in self.graph else self.graph[target] + [gps]
        if self.ds_right_front.getValue() > 0.12 and self.ds_right_back.getValue() > 0.12 and self.is_not_visited(270) and not self.is_hole(self.camera_right, 'right'):
            target = (gps[0] - self.tile * self.axes[((heading + 270) % 360)][0], gps[1] - self.tile * self.axes[((heading + 270) % 360)][1])
            self.not_visited.append([target[0], target[1]])
            self.graph[gps].append(target)
            self.graph[target] = [gps] if target not in self.graph else self.graph[target] + [gps]
        if self.ds_left_front.getValue() > 0.12 and self.ds_left_back.getValue() > 0.12 and self.is_not_visited(90) and not self.is_hole(self.camera_left, 'left'):
            target = (gps[0] - self.tile * self.axes[((heading + 90) % 360)][0], gps[1] - self.tile * self.axes[((heading + 90) % 360)][1])
            self.not_visited.append([target[0], target[1]])
            self.graph[gps].append(target)
            self.graph[target] = [gps] if target not in self.graph else self.graph[target] + [gps]
        if self.ds_right_front.getValue() > 0.12 and self.ds_right_back.getValue() <= 0.12 and self.ds_front_right.getValue() > 0.12 and self.ds_front_left.getValue() <= 0.12 and self.is_not_visited(315):
            target = (gps[0] - self.tile * self.axes[((heading + 315) % 360)][0] * 2, gps[1] - self.tile * self.axes[((heading + 315) % 360)][1] * 2)
            self.graph[gps].append(target)
            self.graph[target] = [gps] if target not in self.graph else self.graph[target] + [gps]
        if self.ds_left_front.getValue() > 0.12 and self.ds_left_back.getValue() <= 0.12 and self.ds_front_left.getValue() > 0.12 and self.ds_front_right.getValue() <= 0.12 and self.is_not_visited(45):
            target = (gps[0] - self.tile * self.axes[((heading + 45) % 360)][0] * 2, gps[1] - self.tile * self.axes[((heading + 45) % 360)][1] * 2)
            self.graph[gps].append(target)
            self.graph[target] = [gps] if target not in self.graph else self.graph[target] + [gps]
    
    def find_closest_not_visited_position(self, target):
        gps = (lambda x: (round(x[0] / self.tile, 0) * self.tile, round(x[1] / self.tile, 0) * self.tile))(self.get_gps_coords(3))
        paths = [[gps]]
        a = 0
        while True and a <= 100:
            a += 1
            new_paths = []
            for path in paths:
                for pos in self.graph[path[-1]]:
                    if [pos[0], pos[1]] in target:
                        return path + [pos]
                    if pos not in path and pos not in [x[-1] for x in new_paths] and pos not in self.wall_positions:
                        new_paths.append(path + [pos])
            paths = new_paths.copy()
        self.send_map()
        self.delay(1500)
        self.emitter.send(bytes('E', "utf-8"))
    
    def get_to_position(self, path, finish):
        print(path)
        for pos_idx in range(len(path)-1):
            angle = (round(atan2(path[pos_idx][0] - path[pos_idx+1][0], path[pos_idx][1] - path[pos_idx+1][1]) * 360 / pi / 45, 0) * 22.5) % 360 - 180
            angle = (angle + 360 - self.get_heading()) % 360 - 180
            #self.rotation(angle if angle != -180 or (pos_idx == len(path) - 2 and angle == -180 and not finish) else 0)
            self.rotation(angle)
            if pos_idx != len(path) - 2 or finish:
                #if (self.ds_front_left.getValue() > 0.12 and self.ds_front_centre_left.getValue() > 0.12  and self.ds_front_centre_right.getValue() > 0.12 and self.ds_front_right.getValue() > 0.12) or angle == -180:
                if (self.ds_front_left.getValue() > 0.12 and self.ds_front_centre_left.getValue() > 0.12  and self.ds_front_centre_right.getValue() > 0.12 and self.ds_front_right.getValue() > 0.12):
                    #self.movement((0.5 if angle != -180 else -0.5) if angle % 90 == 0 else 1)
                    self.movement(0.5 if angle % 90 == 0 else 1)
                else:
                    self.graph[path[0]].remove(path[1])
                    self.graph[path[1]].remove(path[0])
                    return
            else:
                self.not_visited = [i for i in self.not_visited if i != [path[-1][0], path[-1][1]]]

    def wall_update(self):
        gps = (lambda x: [int(round(x[0] / self.tile, 0) * self.tile), int(round(x[1] / self.tile, 0) * self.tile)])(self.get_gps_coords(3))
        heading = self.get_heading()

        if gps[0] % 12 == self.rests[0] and gps[1] % 12 == self.rests[1] and heading % 90 == 0:
            if self.ds_front_left.getValue() < 0.12 or self.ds_left_front.getValue() < 0.12:
                self.wall_positions.append((gps[0] - self.axes[(heading + 45) % 360][0] * self.tile, gps[1] - self.axes[(heading + 45) % 360][1] * self.tile))
                self.visited.append((gps[0] - self.axes[(heading + 45) % 360][0] * self.tile, gps[1] - self.axes[(heading + 45) % 360][1] * self.tile))
            if self.ds_front_right.getValue() < 0.12 or self.ds_right_front.getValue() < 0.12:
                self.wall_positions.append((gps[0] - self.axes[(heading + 315) % 360][0] * self.tile, gps[1] - self.axes[(heading + 315) % 360][1] * self.tile))
                self.visited.append((gps[0] - self.axes[(heading + 315) % 360][0] * self.tile, gps[1] - self.axes[(heading + 315) % 360][1] * self.tile))
            if self.ds_front_centre_left.getValue() < 0.12 or self.ds_front_centre_right.getValue() < 0.12:
                self.wall_positions.append((gps[0] - self.axes[heading][0] * self.tile, gps[1] - self.axes[heading][1] * self.tile))
                self.visited.append((gps[0] - self.axes[heading][0] * self.tile, gps[1] - self.axes[heading][1] * self.tile))
            if self.ds_left_back.getValue() < 0.12 and self.ds_left_front.getValue() < 0.12:
                self.wall_positions.append((gps[0] - self.axes[(heading + 90) % 360][0] * self.tile, gps[1] - self.axes[(heading + 90) % 360][1] * self.tile))
                self.visited.append((gps[0] - self.axes[(heading + 90) % 360][0] * self.tile, gps[1] - self.axes[(heading + 90) % 360][1] * self.tile))
            if self.ds_left_back.getValue() < 0.12:
                self.wall_positions.append((gps[0] - self.axes[(heading + 135) % 360][0] * self.tile, gps[1] - self.axes[(heading + 135) % 360][1] * self.tile))
                self.visited.append((gps[0] - self.axes[(heading + 135) % 360][0] * self.tile, gps[1] - self.axes[(heading + 135) % 360][1] * self.tile))
            if self.ds_right_back.getValue() < 0.12 and self.ds_right_front.getValue() < 0.12:
                self.wall_positions.append((gps[0] - self.axes[(heading + 270) % 360][0] * self.tile, gps[1] - self.axes[(heading + 270) % 360][1] * self.tile))    
                self.visited.append((gps[0] - self.axes[(heading + 270) % 360][0] * self.tile, gps[1] - self.axes[(heading + 270) % 360][1] * self.tile))    
            if self.ds_right_back.getValue() < 0.12:
                self.visited.append((gps[0] - self.axes[(heading + 225) % 360][0] * self.tile, gps[1] - self.axes[(heading + 225) % 360][1] * self.tile))     
                self.wall_positions.append((gps[0] - self.axes[(heading + 225) % 360][0] * self.tile, gps[1] - self.axes[(heading + 225) % 360][1] * self.tile))     

    def get_gps_coords(self, r):
        return (lambda x: [round(x[0]*100, r), round(x[2]*100, r)])(self.gps.getValues())

    def get_gyro_angle(self):
        self.angle +=  (self.timestep / 1000 ) * self.gyro.getValues()[1]
        return self.angle/pi*180
    
    def get_heading(self):
        return int(round(((self.angle / pi * 180) % 360 + 360) % 360 / 45, 0) % 8 * 45)

    def is_not_visited(self, turn):
        pos = (lambda x, y: [round((x[0] - self.axes[(y + turn) % 360][0] * self.tile) / self.tile, 0) * self.tile, round((x[1] - self.axes[(y + turn) % 360][1] * self.tile) / self.tile, 0) * self.tile])(self.get_gps_coords(3), self.get_heading())
        return pos not in self.visited and pos not in self.wall_positions

    def add_visited(self):
        pos = (lambda x, y: [round((x[0] - self.axes[(y) % 360][0] * self.tile) / self.tile, 0) * self.tile, round((x[1] - self.axes[(y) % 360][1] * self.tile) / self.tile, 0) * self.tile])(self.get_gps_coords(3), self.get_heading())
        self.visited.append(pos)
        self.extend_visited(pos)
    
    def extend_visited(self, gps):
        if [gps[0] - self.tile, gps[1]] not in self.visited and [gps[0] - 2 * self.tile, gps[1]] in self.visited and ([gps[0] - self.tile, gps[1] - self.tile] in self.visited or [gps[0] - self.tile, gps[1] + self.tile] in self.visited):
            self.visited.append([gps[0] - self.tile, gps[1]])
            self.extend_visited([gps[0] - self.tile, gps[1]])
            self.extend_graph([gps[0] - self.tile, gps[1]])
        elif [gps[0] + self.tile, gps[1]] not in self.visited and [gps[0] + 2 * self.tile, gps[1]] in self.visited and ([gps[0] + self.tile, gps[1] - self.tile] in self.visited or [gps[0] + self.tile, gps[1] + self.tile] in self.visited):
            self.visited.append([gps[0] + self.tile, gps[1]])
            self.extend_visited([gps[0] + self.tile, gps[1]])
            self.extend_graph([gps[0] + self.tile, gps[1]])
        elif [gps[0], gps[1] - self.tile] not in self.visited and [gps[0], gps[1] - 2 * self.tile] in self.visited and ([gps[0] + self.tile, gps[1] - self.tile] in self.visited or [gps[0] - self.tile, gps[1] - self.tile] in self.visited):
            self.visited.append([gps[0], gps[1] - self.tile])
            self.extend_visited([gps[0], gps[1] - self.tile])
            self.extend_graph([gps[0], gps[1] - self.tile])
        elif [gps[0], gps[1] + self.tile] not in self.visited and [gps[0], gps[1] + 2 * self.tile] in self.visited and ([gps[0] + self.tile, gps[1] + self.tile] in self.visited or [gps[0] - self.tile, gps[1] + self.tile] in self.visited):
            self.visited.append([gps[0], gps[1] + self.tile])
            self.extend_visited([gps[0], gps[1] + self.tile])
            self.extend_graph([gps[0], gps[1] + self.tile])
    
    def extend_graph(self, gps):
        self.graph[(gps[0], gps[1])] = []
        if [gps[0] - self.tile, gps[1]] in self.visited and [gps[0] - self.tile, gps[1]] not in self.wall_positions:
            if (gps[0] - self.tile, gps[1]) not in self.graph:
                self.graph[(gps[0] - self.tile, gps[1])] = []
            self.graph[(gps[0] - self.tile, gps[1])].append((gps[0], gps[1]))
            self.graph[(gps[0], gps[1])].append((gps[0] - self.tile, gps[1]))
        if [gps[0] + self.tile, gps[1]] in self.visited and [gps[0] + self.tile, gps[1]] not in self.wall_positions:
            if (gps[0] + self.tile, gps[1]) not in self.graph:
                self.graph[(gps[0] + self.tile, gps[1])] = []
            self.graph[(gps[0] + self.tile, gps[1])].append((gps[0], gps[1]))
            self.graph[(gps[0], gps[1])].append((gps[0] + self.tile, gps[1]))
        if [gps[0], gps[1] - self.tile] in self.visited and [gps[0], gps[1] - self.tile] not in self.wall_positions:
            if (gps[0], gps[1] - self.tile) not in self.graph:
                self.graph[(gps[0], gps[1] - self.tile)] = []
            self.graph[(gps[0], gps[1] - self.tile)].append((gps[0], gps[1]))
            self.graph[(gps[0], gps[1])].append((gps[0], gps[1] - self.tile))
        if [gps[0], gps[1] + self.tile] in self.visited and [gps[0], gps[1] + self.tile] not in self.wall_positions:
            if (gps[0], gps[1] + self.tile) not in self.graph:
                self.graph[(gps[0], gps[1] + self.tile)] = []
            self.graph[(gps[0], gps[1] + self.tile)].append((gps[0], gps[1]))
            self.graph[(gps[0], gps[1])].append((gps[0], gps[1] + self.tile))

    def mapping(self):
        gps = (lambda x: [int(round(x[0] / self.tile, 0) * self.tile), int(round(x[1] / self.tile, 0) * self.tile)])(self.get_gps_coords(3))

        if gps[0] % 12 == self.rests[0] and gps[1] % 12 == self.rests[1] and self.get_surface() != 2 and self.get_heading() % 90 == 0:
            self.map_pos[0] = int((self.map_pos[0] + 4 * ((gps[0] - self.map_gps[0]) // (self.tile * 2))))
            self.map_pos[1] = int(self.map_pos[1] + 4 * ((gps[1] - self.map_gps[1]) // (self.tile * 2)))

            if self.map_pos[0] <= 1:
                self.map_matrix = [['*' for _ in range((2 - self.map_pos[0]))] + i for i in self.map_matrix]
                self.map_pos[0] += 2 - self.map_pos[0]
            if self.map_pos[0] >= len(self.map_matrix[0]) - 2:
                self.map_matrix = [i + ['*' for _ in range((3 - len(self.map_matrix[0]) + self.map_pos[0]))] for i in self.map_matrix]
            if self.map_pos[1] <= 1:
                self.map_matrix = [['*' for _ in range(len(self.map_matrix[0]))] for _ in range(2 - self.map_pos[1])] + self.map_matrix
                self.map_pos[1] += 2 - self.map_pos[1]                   
            if self.map_pos[1] >= len(self.map_matrix) - 2:
                self.map_matrix = self.map_matrix + [['*' for _ in range(len(self.map_matrix[0]))] for _ in range(3 - len(self.map_matrix) + self.map_pos[1])]                   


            surface = self.get_surface()
            if self.map_matrix[self.map_pos[1]][self.map_pos[0]] == '*':
                self.map_matrix[self.map_pos[1] - 1][self.map_pos[0] - 1:self.map_pos[0] + 2] = [str(surface), '0', str(surface)]
                self.map_matrix[self.map_pos[1]][self.map_pos[0] - 1:self.map_pos[0] + 2] = ['0', '0', '0']
                self.map_matrix[self.map_pos[1] + 1][self.map_pos[0] - 1:self.map_pos[0] + 2] = [str(surface), '0', str(surface)]

            h = self.get_heading()
            self.map_matrix[self.map_pos[1] + self.map_axes[h]['l'][1]][self.map_pos[0] + self.map_axes[h]['l'][0]] = '1' if 0.06 < self.ds_front_left.getValue() <= 0.12 or self.map_matrix[self.map_pos[1] + self.map_axes[h]['l'][1]][self.map_pos[0] + self.map_axes[h]['l'][0]] == '1' else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[h]['cl'][1]][self.map_pos[0] + self.map_axes[h]['cl'][0]] = '1' if 0.06 < self.ds_front_left.getValue() <= 0.12 or self.map_matrix[self.map_pos[1] + self.map_axes[h]['cl'][1]][self.map_pos[0] + self.map_axes[h]['cl'][0]] == '1'else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[h]['c'][1]][self.map_pos[0] + self.map_axes[h]['c'][0]] = '1' if 0.06 < self.ds_front_centre_left.getValue() <= 0.12 or self.ds_front_centre_right.getValue() <= 0.12 else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[h]['cr'][1]][self.map_pos[0] + self.map_axes[h]['cr'][0]] = '1' if 0.06 < self.ds_front_right.getValue() <= 0.12 or self.map_matrix[self.map_pos[1] + self.map_axes[h]['cr'][1]][self.map_pos[0] + self.map_axes[h]['cr'][0]] == '1' else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[h]['r'][1]][self.map_pos[0] + self.map_axes[h]['r'][0]] = '1' if 0.06 < self.ds_front_right.getValue() <= 0.12 or self.map_matrix[self.map_pos[1] + self.map_axes[h]['r'][1]][self.map_pos[0] + self.map_axes[h]['r'][0]] == '1' else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[h]['bl'][1]][self.map_pos[0] + self.map_axes[h]['bl'][0]] = '1' if self.ds_front_left.getValue() <= 0.06 else self.map_matrix[self.map_pos[1] + self.map_axes[h]['bl'][1]][self.map_pos[0] + self.map_axes[h]['bl'][0]]
            self.map_matrix[self.map_pos[1] + self.map_axes[h]['br'][1]][self.map_pos[0] + self.map_axes[h]['br'][0]] = '1' if self.ds_front_right.getValue() <= 0.06 else self.map_matrix[self.map_pos[1] + self.map_axes[h]['br'][1]][self.map_pos[0] + self.map_axes[h]['br'][0]]
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 90) % 360]['l'][1]][self.map_pos[0] + self.map_axes[(h + 90) % 360]['l'][0]] = '1' if 0.06 < self.ds_left_back.getValue() <= 0.12 or self.map_matrix[self.map_pos[1] + self.map_axes[(h + 90) % 360]['l'][1]][self.map_pos[0] + self.map_axes[(h + 90) % 360]['l'][0]] == '1' else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 90) % 360]['cl'][1]][self.map_pos[0] + self.map_axes[(h + 90) % 360]['cl'][0]] = '1' if 0.06 < self.ds_left_back.getValue() <= 0.12 or self.map_matrix[self.map_pos[1] + self.map_axes[(h + 90) % 360]['cl'][1]][self.map_pos[0] + self.map_axes[(h + 90) % 360]['cl'][0]] == '1' else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 90) % 360]['c'][1]][self.map_pos[0] + self.map_axes[(h + 90) % 360]['c'][0]] = '1' if 0.06 < self.ds_left_back.getValue() <= 0.12 or self.ds_left_front.getValue() <= 0.12 else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 90) % 360]['cr'][1]][self.map_pos[0] + self.map_axes[(h + 90) % 360]['cr'][0]] = '1' if 0.06 < self.ds_left_front.getValue() <= 0.12 or self.map_matrix[self.map_pos[1] + self.map_axes[(h + 90) % 360]['cr'][1]][self.map_pos[0] + self.map_axes[(h + 90) % 360]['cr'][0]] == '1' else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 90) % 360]['r'][1]][self.map_pos[0] + self.map_axes[(h + 90) % 360]['r'][0]] = '1' if 0.06 < self.ds_left_front.getValue() <= 0.12 or self.map_matrix[self.map_pos[1] + self.map_axes[(h + 90) % 360]['r'][1]][self.map_pos[0] + self.map_axes[(h + 90) % 360]['r'][0]] == '1' else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 90) % 360]['bl'][1]][self.map_pos[0] + self.map_axes[(h + 90) % 360]['bl'][0]] = '1' if self.ds_left_back.getValue() <= 0.06 else self.map_matrix[self.map_pos[1] + self.map_axes[(h + 90) % 360]['bl'][1]][self.map_pos[0] + self.map_axes[(h + 90) % 360]['bl'][0]]
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 90) % 360]['br'][1]][self.map_pos[0] + self.map_axes[(h + 90) % 360]['br'][0]] = '1' if self.ds_left_front.getValue() <= 0.06 else self.map_matrix[self.map_pos[1] + self.map_axes[(h + 90) % 360]['br'][1]][self.map_pos[0] + self.map_axes[(h + 90) % 360]['br'][0]]
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 270) % 360]['l'][1]][self.map_pos[0] + self.map_axes[(h + 270) % 360]['l'][0]] = '1' if 0.06 < self.ds_right_front.getValue() <= 0.12 or self.map_matrix[self.map_pos[1] + self.map_axes[(h + 270) % 360]['l'][1]][self.map_pos[0] + self.map_axes[(h + 270) % 360]['l'][0]] == '1' else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 270) % 360]['cl'][1]][self.map_pos[0] + self.map_axes[(h + 270) % 360]['cl'][0]] = '1' if 0.06 < self.ds_right_front.getValue() <= 0.12 or self.map_matrix[self.map_pos[1] + self.map_axes[(h + 270) % 360]['cl'][1]][self.map_pos[0] + self.map_axes[(h + 270) % 360]['cl'][0]] == '1' else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 270) % 360]['c'][1]][self.map_pos[0] + self.map_axes[(h + 270) % 360]['c'][0]] = '1' if 0.06 < self.ds_right_front.getValue() <= 0.12 or self.ds_right_back.getValue() <= 0.12 else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 270) % 360]['cr'][1]][self.map_pos[0] + self.map_axes[(h + 270) % 360]['cr'][0]] = '1' if 0.06 < self.ds_right_back.getValue() <= 0.12 or self.map_matrix[self.map_pos[1] + self.map_axes[(h + 270) % 360]['cr'][1]][self.map_pos[0] + self.map_axes[(h + 270) % 360]['cr'][0]] == '1' else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 270) % 360]['r'][1]][self.map_pos[0] + self.map_axes[(h + 270) % 360]['r'][0]] = '1' if 0.06 < self.ds_right_back.getValue() <= 0.12 or self.map_matrix[self.map_pos[1] + self.map_axes[(h + 270) % 360]['r'][1]][self.map_pos[0] + self.map_axes[(h + 270) % 360]['r'][0]] == '1' else '0'
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 270) % 360]['bl'][1]][self.map_pos[0] + self.map_axes[(h + 270) % 360]['bl'][0]] = '1' if self.ds_right_front.getValue() <= 0.06 else self.map_matrix[self.map_pos[1] + self.map_axes[(h + 270) % 360]['bl'][1]][self.map_pos[0] + self.map_axes[(h + 270) % 360]['bl'][0]]
            self.map_matrix[self.map_pos[1] + self.map_axes[(h + 270) % 360]['br'][1]][self.map_pos[0] + self.map_axes[(h + 270) % 360]['br'][0]] = '1' if self.ds_right_back.getValue() <= 0.06 else self.map_matrix[self.map_pos[1] + self.map_axes[(h + 270) % 360]['br'][1]][self.map_pos[0] + self.map_axes[(h + 270) % 360]['br'][0]]

            self.map_gps = gps

    def get_remaining_time(self):
        message = struct.pack('c', 'G'.encode())
        self.emitter.send(message)

        if self.receiver.getQueueLength() > 0:
            receivedData = self.receiver.getBytes()
            try:
                tup = struct.unpack('c f i', receivedData)
            except Exception:
                return 480
            if tup[0].decode("utf-8") == 'G':
                self.receiver.nextPacket() 
                return tup[2]
        return 480

    def is_lack_of_progress(self):
        if self.receiver.getQueueLength() > 0: # If receiver queue is not empty
            receivedData = self.receiver.getBytes()
            try:
                tup = struct.unpack('c', receivedData) # Parse data into character
            except Exception:
                return False
            if tup[0].decode("utf-8") == 'L': # 'L' means lack of progress occurred
                self.send_map()
                self.delay(1500)
                self.send_map()
                return True

            self.receiver.nextPacket()
        return False

    def send_map(self):
        #self.map_matrix = [[x if x != '*' else '0' for x in i] for i in self.map_matrix]
        self.map_matrix = [['1' if (i == 0 or i == len(self.map_matrix)-1 or x == 0 or x == len(self.map_matrix[0])-1) else (self.map_matrix[i][x] if self.map_matrix[i][x] != '*' else '0') for x in range(len(self.map_matrix[0]))] for i in range(len(self.map_matrix))]
        map_matrix = np.array(self.map_matrix)
        s = map_matrix.shape
        s_bytes = struct.pack('2i',*s)

        flatMap = ','.join(map_matrix.flatten())
        sub_bytes = flatMap.encode('utf-8')

        a_bytes = s_bytes + sub_bytes

        self.emitter.send(a_bytes)

        map_evaluate_request = struct.pack('c', b'M')
        self.emitter.send(map_evaluate_request)

        exit_mes = struct.pack('c', b'E')
        self.emitter.send(exit_mes)
        print('===')
        for i in self.map_matrix:
            print(*i, sep='')
        print('===')

    #-----------------------------------camera---------------

    def send(self, info, x, y):
        for i in self.victim_locations:
            if x in range(i[0]-7,i[0]+7) and y in range(i[1]-7,i[1]+7):
                break
        else:
            self.wheel_left.setVelocity(0)
            self.wheel_right.setVelocity(0)
            print(info)
            victimType = bytes(info, "utf-8")
            message = struct.pack("i i c", x, y, victimType) # Pack the message.
            self.emitter.send(message)
            self.delay(1500)
            self.emitter.send(message)
            self.delay(1500)
            self.victim_locations.append([x,y])
    
    def delay(self, ms):
        initTime = self.robot.getTime()      # Store starting time (in seconds)
        while self.robot.step(self.timestep) != -1:
            if (self.robot.getTime() - initTime) * 1000.0 > ms: # If time elapsed (converted into ms) is greater than value passed in
                break
            if self.get_remaining_time() <= 3 and not self.sent:
                self.send_map()
                self.sent = True
    
    def createBWimage(self, inputPathImage):
        im_gray = cv.imread(inputPathImage, cv.IMREAD_GRAYSCALE)
        self.im_bw = cv.threshold(im_gray, 180, 255, cv.THRESH_BINARY)[1]
    
    def cropImage(self, inputImage):
        cnts = cv.findContours(inputImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)

        for c in cnts:
            x,y,w,h = cv.boundingRect(c)
            self.ROI = inputImage[y:y+h, x:x+w]
            break
    
    def letterDetection(self, input_array, pos):
        x = len(input_array[0]) // 2
        y = len(input_array) // 2
        if input_array[y][x] <= 50:
            x = len(input_array[0]) // 2
            y = int(len(input_array) * 0.85)
            if input_array[y][x] <= 50:
                result = 'S'
            else:
                result = 'H'
        else:
            result = 'U'

        self.send(result, pos[0], pos[1])

    def detect_victim(self, camera, camera_pos):
        image_data = camera.getImage()
        img = np.array(np.frombuffer(image_data, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4)))
        img[:,:,2] = np.zeros([img.shape[0], img.shape[1]])

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        thresh = cv.threshold(gray, 140, 255, cv.THRESH_BINARY)[1]
        thresh_white = cv.inRange(img, (140, 120, 0, 0), (210, 210, 30, 255))
        thresh_red = cv.inRange(img, (60, 0, 0, 0), (110, 30, 30, 255))
        thresh_black = cv.inRange(img, (0, 0, 0, 0), (30, 30, 30, 255))

        contours, h = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_white, h = cv.findContours(thresh_white, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_red, h = cv.findContours(thresh_red, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_black, h = cv.findContours(thresh_black, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        pos = [int(i) for i in (lambda x: [round(x[0] / self.tile, 0) * self.tile, round(x[1] / self.tile, 0) * self.tile])(self.get_gps_coords(3))]

        for c in contours_red:
            if cv.contourArea(c) > 30 and all(i.getValue() <= 0.12 for i in self.camera_pos_ds[camera_pos]):
                self.wheel_left.setVelocity(0)
                self.wheel_right.setVelocity(0)

                thresh_yellow = cv.inRange(img, (0, 140, 0, 0), (30, 220, 30, 255))
                contours_yellow, h = cv.findContours(thresh_yellow, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                for c in contours_yellow:
                    if cv.contourArea(c) > 20:
                        self.send('O', pos[0], pos[1])
                        break
                else:
                    self.send('F', pos[0], pos[1])
        
        for c in contours:
            if cv.contourArea(c) > 20 and all(i.getValue() <= 0.12 for i in self.camera_pos_ds[camera_pos]):
                M = cv.moments(c)
                cX = int(M["m10"] / M["m00"])

                if int(len(img[0]) * 0.25) <= cX <= int(len(img[0]) * 0.75):
                    if cv.contourArea(c) > 300:
                        for c in contours_black:
                            print(cv.contourArea(c))
                            if cv.contourArea(c) > 100:
                                self.cropImage(thresh)
                                self.letterDetection(self.ROI, pos)

        for c in contours_white:
            if cv.contourArea(c) >= 20:
                M = cv.moments(c)
                cX = int(M["m10"] / M["m00"])
                #if (800 > cv.contourArea(c) > 500 and self.camera_pos_ds[camera_pos][0].getValue() <= 0.12 and self.camera_pos_ds[camera_pos][1].getValue() <= 0.10 and int(len(img[0]) * 0.25) <= cX <= int(len(img[0]) * 0.75)):
                 #   self.send('P', pos[0], pos[1])
                if (800 > cv.contourArea(c) > 100 and all(i.getValue() <= 0.12 for i in self.camera_pos_ds[camera_pos]) and int(len(img[0]) * 0.25) <= cX <= int(len(img[0]) * 0.75)):
                    
                    for c in contours_black:
                            print(cv.contourArea(c))
                            if cv.contourArea(c) > 100:
                                self.send('C', pos[0], pos[1])
                                return
                    self.send('P', pos[0], pos[1])
        
    def is_hole(self, camera, camera_pos):
        image_data = camera.getImage()
        img = np.array(np.frombuffer(image_data, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4)))
        img[:,:,2] = np.zeros([img.shape[0], img.shape[1]])

        return (img[-1][4][0] in range(0, 25) and img[-1][4][1] in range(0, 25) and img[-1][4][2] in range(0, 25) or img[-1][-4][0] in range(0, 25) and img[-1][-4][1] in range(0, 25) and img[-1][-4][2] in range(0, 25))



class Robot0(ErebusRobot):
    def test(self):
        while self.robot.step(self.timestep) != -1:
            self.wheel_left.setVelocity(0)
            self.wheel_right.setVelocity(0)


    def run(self):
        while self.robot.step(self.timestep) != -1:
            self.wheel_left.setVelocity(0)
            self.wheel_right.setVelocity(0)
            self.while_algorithm()
            self.get_to_position(self.find_closest_not_visited_position([self.visited[0]]), True)
            self.send_map()
            self.delay(1500)
            self.send_map()
            self.delay(1500)
            self.emitter.send(bytes('E', "utf-8"))
            break

    def while_algorithm(self): 
        try:   
            while self.robot.step(self.timestep) != -1:
                self.graph_update()
                self.wall_update()
                self.not_visited = [i for i in self.not_visited if i not in self.visited]
                gps = self.get_gps_coords(3)
                tile_center = (lambda x: (round(x[0] / self.tile, 0) * self.tile, round(x[1] / self.tile, 0) * self.tile))(self.get_gps_coords(3))
                

                if self.get_surface() in ([2, 8, 9] if not self.room4 else [2]):
                    self.movement(-self.passed)
                
                if not tile_center[0] - 0.7 <= gps[0] <= tile_center[0] + 0.7:
                    self.rotation(90)
                    self.movement(0.1 if gps[0] < tile_center[0] else -0.1)
                    self.rotation(-90)
                if not tile_center[1] - 0.7 <= gps[1] <= tile_center[1] + 0.7:
                    self.rotation(90)
                    self.movement(0.1 if gps[1] < tile_center[1] else -0.1)
                    self.rotation(-90)
                
                if (self.ds_front_left.getValue() <= 0.12 or self.ds_front_right.getValue() <= 0.12) and (self.ds_left_back.getValue() <= 0.12 or self.ds_left_front.getValue() <= 0.12) and (self.ds_right_back.getValue() <= 0.12 or self.ds_right_front.getValue() <= 0.12) and not (self.ds_front_left.getValue() > 0.12 and self.ds_left_front.getValue() > 0.12) and not (self.ds_front_right.getValue() > 0.12 and self.ds_right_front.getValue() > 0.12):
                    self.rotation(180)

                if self.ds_front_left.getValue() > 0.12 and self.ds_front_centre_left.getValue() > 0.12  and self.ds_front_centre_right.getValue() > 0.12 and self.ds_front_right.getValue() > 0.12 and self.is_not_visited(0) and not self.is_hole(self.camera_centre, 'centre'):
                    self.add_visited()
                    self.movement(0.5)
                    self.mapping()
                
                elif self.ds_right_front.getValue() > 0.12 and self.ds_right_back.getValue() <= 0.12 and self.ds_front_right.getValue() > 0.12 and self.ds_front_left.getValue() <= 0.12 and self.is_not_visited(315):
                    self.rotation(-45)
                    self.add_visited()
                    self.movement(1)
                    self.rotation(45)

                elif self.ds_left_front.getValue() > 0.12 and self.ds_left_back.getValue() <= 0.12 and self.ds_front_left.getValue() > 0.12 and self.ds_front_right.getValue() <= 0.12 and self.is_not_visited(45):
                    self.rotation(45)
                    self.add_visited()
                    self.movement(1)
                    self.rotation(-45)

                elif self.ds_right_front.getValue() > 0.12 and self.ds_right_back.getValue() > 0.12 and self.is_not_visited(270) and not self.is_hole(self.camera_right, 'right'):
                    self.rotation(-90)
                    self.mapping()

                elif self.ds_left_front.getValue() > 0.12 and self.ds_left_back.getValue() > 0.12 and self.is_not_visited(90) and not self.is_hole(self.camera_left, 'left'):
                    self.rotation(90)
                    self.mapping()

                else:
                    if self.not_visited == []:
                        break
                    else:
                        self.get_to_position(self.find_closest_not_visited_position(self.not_visited), False)
        except Exception:
            self.send_map()


robot0 = Robot0()
robot0.run()




#TO DO: diery                                       done
#       plynule rovne chodenie - diery              done
#       poison
#       45 stupnocve pohyby po polke tilu
#       prepojenia grafov efektivnejsie
#       vsetky positions v tuple
#       optimalizovat djikstraas                    done