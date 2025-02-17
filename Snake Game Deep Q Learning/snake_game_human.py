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
        """
        Place food at a random position on the grid that is not occupied by the snake.
        Uses a loop to avoid recursion.
        """
        while True:
            x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            food = Point(x, y)
            if food not in self.snake:
                self.food = food
                break

    def play_step(self):
        """
        Execute one step in the game.
        Returns:
            game_over (bool): Whether the game has ended.
            score (int): The current score.
        """
        # 1. Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

        # 2. Move the snake
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # 3. Check for self-collision
        if self._is_collision():
            return True, self.score

        # 4. Check if food is eaten
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update the user interface and control the game speed
        self._update_ui()
        self.clock.tick(SPEED)
        return False, self.score

    def _handle_keydown(self, event):
        """
        Handle key presses to update the snake's direction.
        Prevents reversing direction directly.
        """
        if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
            self.direction = Direction.LEFT
        elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
            self.direction = Direction.RIGHT
        elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
            self.direction = Direction.UP
        elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
            self.direction = Direction.DOWN

    def _is_collision(self):
        """Return True if the snake collides with itself."""
        return self.head in self.snake[1:]

    def _update_ui(self):
        """Update the display with the current snake, food, and score."""
        self.display.fill(BLACK)

        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw score
        text = FONT.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, (0, 0))
        pygame.display.flip()

    def _move(self, direction):
        """Move the snake's head in the specified direction, wrapping around the screen."""
        x, y = self.head.x, self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE

        # Wrap around the screen edges
        x = x % self.width
        y = y % self.height
        self.head = Point(x, y)

def main():
    """Main function to run the Snake game."""
    game = SnakeGame()

    while True:
        game_over, score = game.play_step()
        if game_over:
            print("Final Score:", score)
            break

    pygame.quit()

if __name__ == '__main__':
    main()
