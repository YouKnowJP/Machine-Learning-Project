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
from typing import Optional, Tuple, List
import config

pygame.init()
try:
    font = pygame.font.Font('arial.ttf', 25)
except Exception:
    font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = config.BLOCK_SIZE
SPEED = config.SPEED

class SnakeGameAI:
    """Snake game with AI interface."""

    def __init__(self, w: int = 640, h: int = 480) -> None:
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self) -> None:
        """Resets the game state."""
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self) -> None:
        """Places food at a random valid location."""
        while True:
            x = random.randrange(0, self.w, BLOCK_SIZE)
            y = random.randrange(0, self.h, BLOCK_SIZE)
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def play_step(self, action: List[int]) -> Tuple[float, bool, int]:
        self.frame_iteration += 1
        # Calculate Manhattan distance before moving
        old_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 150 * len(self.snake):
            game_over = True
            reward = -1
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 1
            self._place_food()
        else:
            self.snake.pop()
            new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            max_distance = (self.w + self.h) / 2
            reward = (old_distance - new_distance) / max_distance

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt: Optional[Point] = None) -> bool:
        """Checks for collisions."""
        if pt is None:
            pt = self.head
        return pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0 or pt in self.snake[1:]

    def _update_ui(self) -> None:
        """Updates the game display."""
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, (0, 0))
        pygame.display.flip()

    def _move(self, action: List[int]) -> None:
        """Moves the snake based on the action."""
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_index = clock_wise.index(self.direction)
        if sum(action) != 1:
            new_dir = clock_wise[current_index]
        else:
            if action == [1, 0, 0]:
                new_dir = clock_wise[current_index]
            elif action == [0, 1, 0]:
                new_index = (current_index + 1) % 4
                new_dir = clock_wise[new_index]
            elif action == [0, 0, 1]:
                new_index = (current_index - 1) % 4
                new_dir = clock_wise[new_index]
            else:
                new_dir = clock_wise[current_index]
        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_astar_path(self):
        """
        Uses the A* algorithm to find a path from the snake's head to the food.
        The snake's body is treated as an obstacle.
        """
        from astar import astar
        obstacles = set(self.snake)
        path = astar(self.head, self.food, obstacles, self.w, self.h, BLOCK_SIZE)
        return path

if __name__ == '__main__':
    game = SnakeGameAI()
    running = True
    while running:
        action = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        reward, game_over, score = game.play_step(action)
        print(f"Reward: {reward}, Score: {score}")
        if game_over:
            print("Game Over")
            game.reset()
