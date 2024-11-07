import numpy as np
import random

class SteeringOutput:
    def __init__(self, linear=np.array([0, 0]), angular=0):
        self.linear = linear
        self.angular = angular

class Kinematic:
    def __init__(self, position=np.array([0, 0]), orientation=0, velocity=np.array([0, 0]), rotation=0):
        self.position = position
        self.orientation = orientation
        self.velocity = velocity
        self.rotation = rotation

    def update(self, steering, time):
        # Update the position and orientation
        self.position += self.velocity * time
        self.orientation += self.rotation * time
        # Update the velocity and rotation
        self.velocity = steering.linear
        self.rotation = steering.angular
        # Update the orientation based on the new velocity
        self.orientation = self.new_orientation(self.orientation, self.velocity)

    @staticmethod
    def new_orientation(current, velocity):
        # Make sure we have a velocity
        if np.linalg.norm(velocity) > 0:
            # Calculate orientation from the velocity
            return np.arctan2(-velocity[0], velocity[1])
        else:
            # Otherwise use the current orientation
            return current

class KinematicSeek:
    def __init__(self, character, target, maxSpeed):
        self.character = character
        self.target = target
        self.maxSpeed = maxSpeed

    def getSteering(self):
        result = SteeringOutput()
        # Get the direction to the target
        result.linear = self.target.position - self.character.position
        # Normalize the velocity and scale by maxSpeed
        norm = np.linalg.norm(result.linear)
        if norm > 0:
            result.linear = (result.linear / norm) * self.maxSpeed
        else:
            result.linear = np.array([0.0, 0.0])
        # Face in the direction we want to move
        self.character.orientation = Kinematic.new_orientation(self.character.orientation, result.linear)
        result.angular = 0
        return result

class KinematicFlee:
    def __init__(self, character, target, maxSpeed):
        self.character = character
        self.target = target
        self.maxSpeed = maxSpeed

    def getSteering(self):
        result = SteeringOutput()
        # Get the direction to the target
        result.linear =  self.character.position - self.target.position
        # Normalize the velocity and scale by maxSpeed
        result.linear = result.linear / np.linalg.norm(result.linear) * self.maxSpeed
        # Face in the direction we want to move
        self.character.orientation = Kinematic.new_orientation(self.character.orientation, result.linear)
        result.angular = 0
        return result

class KinematicArrive:
    def __init__(self, character, target, maxSpeed, radius, timeToTarget=0.25):
        self.character = character
        self.target = target
        self.maxSpeed = maxSpeed
        self.radius = radius
        self.timeToTarget = timeToTarget

    def getSteering(self):
        result = SteeringOutput()
        # Get the direction to the target
        result.linear = self.target.position - self.character.position
        # Check if we're within the radius
        if np.linalg.norm(result.linear) < self.radius:
            # Request no steering
            return None
        # We need to move to our target, we'd like to get there in timeToTarget seconds
        result.linear = result.linear / self.timeToTarget
        # If this is too fast, clip it to the max speed
        if np.linalg.norm(result.linear) > self.maxSpeed:
            result.linear = result.linear / np.linalg.norm(result.linear) * self.maxSpeed
        # Face in the direction we want to move
        self.character.orientation = Kinematic.new_orientation(self.character.orientation, result.linear)
        result.angular = 0
        return result

class KinematicWander:
    def __init__(self, character, maxSpeed, maxRotation):
        self.character = character
        self.maxSpeed = maxSpeed
        self.maxRotation = maxRotation

    def getSteering(self):
        result = SteeringOutput()
        # Get velocity from the vector form of the orientation
        result.linear = self.maxSpeed * np.array([np.cos(self.character.orientation), np.sin(self.character.orientation)])
        # Change our orientation randomly
        result.angular = self.random_binomial() * self.maxRotation
        return result

    @staticmethod
    def random_binomial():
        return random.uniform(-1, 1)