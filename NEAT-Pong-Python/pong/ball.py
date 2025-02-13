import pygame
import math
import random


class Ball:
    """
    Represents the ball in the Pong game.

    The ball is initialized at a given (x, y) position and moves at a random angle
    (within a specified range) that is never exactly 0, ensuring an interesting trajectory.
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
        self.set_random_velocity()

    def set_random_velocity(self) -> None:
        """
        Set the ball's velocity to a random direction.

        The angle is chosen from (-30, 30) degrees (converted to radians) but
        never 0, and a random horizontal direction is applied.
        """
        angle = self._get_random_angle(-30, 30, excluded=0.0)
        # Randomly choose left or right direction
        direction = random.choice([-1, 1])
        self.x_vel = direction * abs(math.cos(angle) * self.MAX_VEL)
        self.y_vel = math.sin(angle) * self.MAX_VEL

    def _get_random_angle(self, min_angle: float, max_angle: float, excluded: float, tol: float = 1e-5) -> float:
        """
        Generate a random angle (in radians) between min_angle and max_angle (in degrees)
        ensuring that the absolute value of the angle is not within tol of the excluded value.

        :param min_angle: Minimum angle in degrees.
        :param max_angle: Maximum angle in degrees.
        :param excluded: Angle in degrees to avoid (e.g., 0).
        :param tol: Tolerance for exclusion.
        :return: A random angle in radians.
        """
        angle_deg = excluded  # Initialize with the excluded value to enter the loop
        while abs(angle_deg - excluded) < tol:
            angle_deg = random.uniform(min_angle, max_angle)
        return math.radians(angle_deg)

    def draw(self, win: pygame.Surface) -> None:
        """
        Draw the ball on the given pygame window.

        :param win: The pygame surface to draw the ball on.
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

        The ball's horizontal velocity is inverted to send it toward the scoring player.
        """
        self.x = self.original_x
        self.y = self.original_y
        self.set_random_velocity()
        self.x_vel = -self.x_vel  # Reverse horizontal direction
