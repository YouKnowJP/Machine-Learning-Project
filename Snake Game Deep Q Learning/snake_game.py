import pygame
import random
from enum import Enum
from collections import namedtuple

# Initialize Pygame
pygame.init()
# Use system font as fallback if arial.ttf is missing
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
INITIAL_SPEED = 10

# Opposite directions mapping for prevention of reversal
OPPOSITE = {
    Direction.RIGHT: Direction.LEFT,
    Direction.LEFT: Direction.RIGHT,
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
}

class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # Initialize game state
        self.direction = Direction.RIGHT
        # Use integer division to ensure grid alignment
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.speed = INITIAL_SPEED
        self.food = None
        self._place_food()

    def _place_food(self):
        """Place food in a random location not occupied by the snake."""
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            food_point = Point(x, y)
            if food_point not in self.snake:
                self.food = food_point
                break

    def play_step(self) -> (bool, int):
        """Processes one step of the game; returns (game_over, score)."""
        # 1. Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # Process only one key event per frame:
            if event.type == pygame.KEYDOWN:
                self._change_direction(event.key)

        # 2. Move snake
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # 3. Check collision
        if self._is_collision():
            return True, self.score

        # 4. Check if food eaten, update score and adjust speed
        if self.head == self.food:
            self.score += 1
            # Increase speed every few points for dynamic difficulty
            if self.score % 5 == 0:
                self.speed += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update UI and control frame rate
        self._update_ui()
        self.clock.tick(self.speed)
        return False, self.score

    def _change_direction(self, key: int):
        """Change direction based on key press, preventing reversal."""
        new_direction = None
        if key == pygame.K_LEFT:
            new_direction = Direction.LEFT
        elif key == pygame.K_RIGHT:
            new_direction = Direction.RIGHT
        elif key == pygame.K_UP:
            new_direction = Direction.UP
        elif key == pygame.K_DOWN:
            new_direction = Direction.DOWN

        # Ignore if new_direction is opposite of current direction
        if new_direction and new_direction != OPPOSITE[self.direction]:
            self.direction = new_direction

    def _is_collision(self) -> bool:
        """Return True if the snake collides with the wall or itself."""
        # Check boundaries
        if (self.head.x < 0 or self.head.x >= self.w or
            self.head.y < 0 or self.head.y >= self.h):
            return True
        # Check self collision (exclude head itself)
        if self.head in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        """Render the game state to the screen."""
        self.display.fill(BLACK)
        # Draw snake
        for pt in self.snake:
            # Outer rectangle for snake segment
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # Inner rectangle for a nice effect
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, BLOCK_SIZE-8, BLOCK_SIZE-8))
        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # Draw score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction: Direction):
        """Update head position based on current direction."""
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

if __name__ == '__main__':
    game = SnakeGame()

    # Main game loop with a simple game-over restart mechanism
    while True:
        game_over, score = game.play_step()
        if game_over:
            # Display game over message
            game.display.fill(BLACK)
            game_over_text = font.render("Game Over! Score: " + str(score), True, RED)
            game.display.blit(game_over_text, (game.w // 4, game.h // 2))
            pygame.display.flip()
            pygame.time.delay(2000)
            # Restart the game
            game = SnakeGame()
