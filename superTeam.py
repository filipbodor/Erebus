from math import *
from controller import Robot
import cv2 as cv
import numpy as np
import json
import struct
from PIL import Image
import os

class ErebusRobot:
    def __init__(self) -> None:
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.max_velocity = 6.28
        self.rotation = round(1115 / self.timestep, 0)
        self.rotation_right_counter = 0
        self.plus = 0
        self.count = 0

        self.visited = [(0, 0)]
        self.angle = (0, 1)
        self.change = {(0, 1): {'left': (1, 0), 'right': (-1, 0)}, (1, 0): {'left': (0, -1), 'right': (0, 1)}, (0, -1): {'left': (-1, 0), 'right': (1, 0)}, (-1, 0): {'left': (0, 1), 'right': (0, -1)}, }
        self.pos = (0, 0)

        self.graph = {}
        self.not_visited = []

        self.wheel_left = self.robot.getDevice('left motor')
        self.wheel_right = self.robot.getDevice('right motor')
        self.camera = self.robot.getDevice('camera')
        self.receiver = self.robot.getDevice('receiver')
        self.emitter = self.robot.getDevice('emitter')
        self.left_encoder = self.wheel_left.getPositionSensor()
        self.right_encoder = self.wheel_right.getPositionSensor()

        self.left_encoder.enable(self.timestep)
        self.right_encoder.enable(self.timestep)
        self.wheel_left.setPosition(float("inf"))       
        self.wheel_right.setPosition(float("inf"))
        self.camera.enable(self.timestep)
        self.receiver.enable(self.timestep)
    
    def straight(self):         # move one tile forward for the left wall follower algorithm 
        # Set the motor speeds 
        self.wheel_left.setVelocity(2)
        self.wheel_right.setVelocity(2)
        
        # Run the loop for a certain amount of time to move one tile forward
        
        for i in range(int((self.rotation * 2.71))):
            self.robot.step(int(self.timestep))

    def back(self):         # move one tile forward for the left wall follower algorithm 
        # Set the motor speeds 
        self.wheel_left.setVelocity(-2)
        self.wheel_right.setVelocity(-2)
        
        # Run the loop for a certain amount of time to move one tile forward
        
        for i in range(int((self.rotation * 2.71))):
            self.robot.step(int(self.timestep))
    
    def left(self):    #turn left
    # Set the motor speeds to make a 90 degree turn
        self.wheel_left.setVelocity(-2.0)
        self.wheel_right.setVelocity(2.0)
        
        # Run the loop for a certain amount of time to make a 90 degree turn
        for i in range(int(self.rotation)):
            self.robot.step(int(self.timestep))

        self.wheel_left.setVelocity(0.0)
        self.wheel_right.setVelocity(0.0)
    
    def right(self):    #turn left
    # Set the motor speeds to make a 90 degree turn
        self.wheel_left.setVelocity(2.0)
        self.wheel_right.setVelocity(-2.0)
        
        # Run the loop for a certain amount of time to make a 90 degree turn
        for i in range(int(self.rotation)):
            self.robot.step(int(self.timestep))

        self.wheel_left.setVelocity(0.0)
        self.wheel_right.setVelocity(0.0)
    
    def is_hole(self, img):
        return (img[-1][4][0] in range(0, 25) and img[-1][4][1] in range(0, 25) and img[-1][4][2] in range(0, 25) or img[-1][-4][0] in range(0, 25) and img[-1][-4][1] in range(0, 25) and img[-1][-4][2] in range(0, 25))

    def update_graph(self):
        target = (self.pos[0] + self.angle[0], self.pos[1] + self.angle[1])
        if target not in self.graph:
            self.graph[target] = []
        self.graph[self.pos].append(target)
        self.graph[target].append(self.pos)
        

    def is_wall(self, img):

        thresh_dark = cv.inRange(img, (25, 25, 0, 0), (35, 35, 5, 255))
        thresh_light = cv.inRange(img, (130, 120, 0, 0), (140, 130, 5, 255))
        contours_dark, h = cv.findContours(thresh_dark, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_light, h = cv.findContours(thresh_light, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        area = 0

        for c in contours_dark:
            area = cv.contourArea(c)
            if cv.contourArea(c) > 2300:
                return True
    
        for o in contours_light:
            area += cv.contourArea(o)
            if cv.contourArea(o) > 2300:
                return True
        
        if area > 2300:
            return True
        return False
    
    def is_finish(self):
        message = struct.pack('c', 'G'.encode())
        self.emitter.send(message)

        if self.receiver.getQueueLength() > 0:
            receivedData = self.receiver.getBytes()
            try:
                tup = struct.unpack('c f i', receivedData)
                print(tup[1])
                self.receiver.nextPacket()
                if tup[1] > 0:
                    return True
            except Exception:
                return False
            
            if tup[0].decode("utf-8") == 'G': 
                pass
                
        return False
        



class SuperTeamRobot(ErebusRobot):
    def test(self):
        while self.robot.step(self.timestep) != -1:
            self.wheel_left.setVelocity(0)
            self.wheel_right.setVelocity(0)
    
    def run(self):
        while self.robot.step(self.timestep) != -1:
            self.wheel_left.setVelocity(0)
            self.wheel_right.setVelocity(0)
            self.algorithm()
            break
    
    def algorithm(self):
        image_data = self.camera.getImage()
        img = np.array(np.frombuffer(image_data, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4)))
        img[:,:,2] = np.zeros([img.shape[0], img.shape[1]])
        
        print(self.visited)
        print(self.pos)
        print(self.is_finish())

        if self.is_finish():
            self.emitter.send(bytes('E', "utf-8"))
            os.exit()

        if not self.is_hole(img) and not self.is_wall(img) and (self.pos[0] + self.angle[0], self.pos[1] + self.angle[1]) not in self.visited:
            self.straight()
            self.pos = (self.pos[0] + self.angle[0], self.pos[1] + self.angle[1])
            self.visited.append(self.pos)
            self.rotation_right_counter = 0
            self.algorithm()
            self.back()
            self.pos = (self.pos[0] - self.angle[0], self.pos[1] - self.angle[1])

        elif self.rotation_right_counter == 3:
            self.rotation_right_counter = 0
            return
        
        self.rotation_right_counter += 1
        self.right()
        self.angle = self.change[self.angle]['right']
        self.algorithm()
        self.left()
        self.angle = self.change[self.angle]['left']

    def algorithm_(self): 
        while self.robot.step(self.timestep) != -1:
            image_data = self.camera.getImage()
            img = np.array(np.frombuffer(image_data, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4)))
            img[:,:,2] = np.zeros([img.shape[0], img.shape[1]])
            
            #for i in img:
            #    print([[x[0], x[1], x[2]] for x in i])
            #print('######################################')
            #print()


            print(self.visited)
            print(self.pos)
            print((self.pos[0] + self.angle[0], self.pos[1] + self.angle[1]))
            print(self.angle)

            if not self.is_hole(img) and not self.is_wall(img) and (self.pos[0] + self.angle[0], self.pos[1] + self.angle[1]) not in self.visited:
                self.straight()
                #self.update_graph()
                self.pos = (self.pos[0] + self.angle[0], self.pos[1] + self.angle[1])
                self.visited.append(self.pos)
                self.rotation_right_counter = 0
                if self.count == 1:
                    self.plus = 0
                self.count = 1
            elif self.rotation_right_counter == 3:
                self.visited.pop(-2 -self.plus)
                self.plus += 1
                self.rotation_right_counter = 0
                self.count = 0
            else:
                self.rotation_right_counter += 1
                self.right()
                self.angle = self.change[self.angle]['right']
            
            if self.is_finish():
                self.emitter.send(bytes('E', "utf-8"))

            '''
            elif (self.pos[0] + self.change[self.angle]['right'][0], self.pos[1] + self.change[self.angle]['right'][1]) not in self.visited:
                self.right()
                self.angle = self.change[self.angle]['right']
            elif (self.pos[0] + self.change[self.angle]['left'][0], self.pos[1] + self.change[self.angle]['left'][1]) not in self.visited:
                self.left()
                self.angle = self.change[self.angle]['left']'''

superTeamRobot = SuperTeamRobot()
superTeamRobot.run()