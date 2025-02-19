#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:11:36 2025

@author: youknowjp
"""

import pygame
import random
from enum import Enum
from collections import namedtuple

# Initialize pygame
pygame.init()

# Global constants
BLOCK_SIZE = 20
SPEED = 15
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Use SysFont for broader compatibility
FONT = pygame.font.SysFont('arial', 25)


# Define direction enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Define a point for snake parts and food
Point = namedtuple('Point', 'x y')


class SnakeGame:
    """A simple Snake game using pygame with wall wrapping."""

    def __init__(self, width=640, height=480):
        """Initialize the game state and set up the display."""
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Reset the game state to start a new game."""
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self._place_food()

    def _place_food(self):
        """Place food at a random position on the grid, avoiding snake body."""
        while True:
            x = random.randrange(0, self.width, BLOCK_SIZE)
            y = random.randrange(0, self.height, BLOCK_SIZE)
            food = Point(x, y)
            if food not in self.snake:
                self.food = food
                break

    def play_step(self):
        """Execute one step in the game."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True, self.score  # Indicate game over
            if event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

        self._move(self.direction)
        self.snake.insert(0, self.head)

        if self._is_collision():
            return True, self.score

        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return False, self.score

    def _handle_keydown(self, event):
        """Handle key presses for direction changes."""
        new_direction = self.direction
        if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
            new_direction = Direction.LEFT
        elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
            new_direction = Direction.RIGHT
        elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
            new_direction = Direction.UP
        elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
            new_direction = Direction.DOWN
        self.direction = new_direction

    def _is_collision(self):
        """Check for collisions with the snake's body."""
        return self.head in self.snake[1:]

    def _update_ui(self):
        """Update the display."""
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = FONT.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, (0, 0))
        pygame.display.flip()

    def _move(self, direction):
        """Move the snake's head, wrapping around the screen."""
        x, y = self.head.x, self.head.y
        if direction == Direction.RIGHT:
            x = (x + BLOCK_SIZE) % self.width
        elif direction == Direction.LEFT:
            x = (x - BLOCK_SIZE) % self.width
        elif direction == Direction.UP:
            y = (y - BLOCK_SIZE) % self.height
        elif direction == Direction.DOWN:
            y = (y + BLOCK_SIZE) % self.height
        self.head = Point(x, y)


def main():
    """Main function to run the Snake game."""
    game = SnakeGame()
    while True:
        game_over, score = game.play_step()
        if game_over:
            print(f"Final Score: {score}")
            break
    pygame.quit()


if __name__ == '__main__':
    main()
