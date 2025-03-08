import pygame
import math
import random

class Ball:
    """
    Represents the ball in the Pong game.
    """
    MAX_VEL = 5
    RADIUS = 7

    def __init__(self, x: int, y: int) -> None:
        """
        Initialize the ball with its starting position and a random velocity.
        """
        self.original_x = x
        self.original_y = y
        self.x = x
        self.y = y

        # Initialize velocities here
        self.x_vel = 0
        self.y_vel = 0

        self.set_random_velocity()

    def set_random_velocity(self) -> None:
        """
        Set the ball's velocity to a random direction.
        """
        angle = self._get_random_angle(-30, 30, excluded=0.0)
        direction = random.choice([-1, 1])
        self.x_vel = direction * abs(math.cos(angle) * self.MAX_VEL)
        self.y_vel = math.sin(angle) * self.MAX_VEL

    @staticmethod
    def _get_random_angle(min_angle: float, max_angle: float, excluded: float, tol: float = 1e-5) -> float:
        """
        Generate a random angle (in radians) between min_angle and max_angle (in degrees),
        avoiding the 'excluded' angle within a given tolerance.
        """
        angle_deg = excluded
        while abs(angle_deg - excluded) < tol:
            angle_deg = random.uniform(min_angle, max_angle)
        return math.radians(angle_deg)

    def draw(self, win: pygame.Surface) -> None:
        """
        Draw the ball on the given pygame window.
        """
        pygame.draw.circle(win, (255, 255, 255), (int(self.x), int(self.y)), self.RADIUS)

    def move(self) -> None:
        """
        Update the ball's position based on its current velocity.
        """
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self) -> None:
        """
        Reset the ball to its original position and assign a new random velocity.
        """
        self.x = self.original_x
        self.y = self.original_y
        self.set_random_velocity()
        self.x_vel = -self.x_vel
