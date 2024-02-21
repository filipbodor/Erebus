from controller import Robot
from controller import Motor
from controller import PositionSensor

robot = Robot() # Create robot object
timeStep = 32   # timeStep = number of milliseconds between world updates

wheel_left = robot.getDevice("left")    # Motor initialization
wheel_right = robot.getDevice("right")
wheel_left.setPosition(float('inf'))
wheel_right.setPosition(float('inf'))

leftEncoder = wheel_left.getPositionSensor()    # Encoder initialization
rightEncoder = wheel_right.getPositionSensor()
leftEncoder.enable(timeStep)
rightEncoder.enable(timeStep)

wheel_left.setVelocity(5.0)
wheel_right.setVelocity(5.0)

while robot.step(timeStep) != -1:
    if leftEncoder.getValue() > 20.0:   # If left motor has spun more than 20 radians
        wheel_left.setVelocity(0.0)
        wheel_right.setVelocity(0.0)

        break