import pygame
from typing import Tuple


class Paddle:
    VEL: int = 4
    WIDTH: int = 20
    HEIGHT: int = 100

    def __init__(self, x: int, y: int, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        """
        Initialize a Paddle instance.

        :param x: The initial x-coordinate.
        :param y: The initial y-coordinate.
        :param color: The RGB color of the paddle (default is white).
        """
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.color = color

    def draw(self, win: pygame.Surface) -> None:
        """
        Draws the paddle on the given window with a shadow effect.

        :param win: The pygame Surface to draw the paddle on.
        """
        # Draw a shadow for a subtle 3D effect
        shadow_color = (50, 50, 50)
        shadow_offset = 3
        shadow_rect = pygame.Rect(self.x + shadow_offset, self.y + shadow_offset, self.WIDTH, self.HEIGHT)
        pygame.draw.rect(win, shadow_color, shadow_rect)

        # Draw the main paddle
        paddle_rect = pygame.Rect(self.x, self.y, self.WIDTH, self.HEIGHT)
        pygame.draw.rect(win, self.color, paddle_rect)

    def move(self, up: bool = True) -> None:
        """
        Moves the paddle up or down based on the provided direction.

        :param up: If True, move upward; otherwise, move downward.
        """
        if up:
            self.y -= self.VEL
        else:
            self.y += self.VEL

    def reset(self) -> None:
        """
        Resets the paddle to its original position.
        """
        self.x = self.original_x
        self.y = self.original_y
