from .paddle import Paddle
from .ball import Ball
import pygame
from dataclasses import dataclass

pygame.init()

@dataclass
class GameInformation:
    """Holds the game statistics."""
    left_hits: int
    right_hits: int
    left_score: int
    right_score: int

class Game:
    """
    Manages the state, drawing, and updates of the Pong game.
    """
    SCORE_FONT = pygame.font.SysFont("comicsans", 50)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    def __init__(self, window: pygame.Surface, window_width: int, window_height: int) -> None:
        self.window_width = window_width
        self.window_height = window_height
        self.window = window

        self.left_paddle = Paddle(10, self.window_height // 2 - Paddle.HEIGHT // 2)
        self.right_paddle = Paddle(
            self.window_width - 10 - Paddle.WIDTH,
            self.window_height // 2 - Paddle.HEIGHT // 2
        )
        self.ball = Ball(self.window_width // 2, self.window_height // 2)

        self.left_score = 0
        self.right_score = 0
        self.left_hits = 0
        self.right_hits = 0

    def _draw_score(self) -> None:
        """Renders the scores of both players onto the game window."""
        left_score_text = self.SCORE_FONT.render(f"{self.left_score}", True, self.WHITE)
        right_score_text = self.SCORE_FONT.render(f"{self.right_score}", True, self.WHITE)
        self.window.blit(
            left_score_text,
            (self.window_width // 4 - left_score_text.get_width() // 2, 20)
        )
        self.window.blit(
            right_score_text,
            (self.window_width * 3 // 4 - right_score_text.get_width() // 2, 20)
        )

    def _draw_hits(self) -> None:
        """Renders the total hit count onto the game window."""
        hits_text = self.SCORE_FONT.render(f"{self.left_hits + self.right_hits}", True, self.RED)
        self.window.blit(
            hits_text,
            (self.window_width // 2 - hits_text.get_width() // 2, 10)
        )

    def _draw_divider(self) -> None:
        """Draws a dotted divider in the center of the screen."""
        divider_width = 10
        divider_height = self.window_height // 20
        divider_x = self.window_width // 2 - divider_width // 2
        for i in range(10, self.window_height, divider_height):
            if (i // divider_height) % 2 == 0:
                pygame.draw.rect(
                    self.window, self.WHITE, (divider_x, i, divider_width, divider_height)
                )

    def _reflect_ball(self, ball: Ball, paddle: Paddle) -> None:
        """
        Handles the reflection of the ball when it collides with a paddle.
        """
        ball.x_vel *= -1
        # Calculate reflection based on where the ball hit the paddle
        middle_y = paddle.y + Paddle.HEIGHT / 2
        difference_in_y = middle_y - ball.y
        reduction_factor = (Paddle.HEIGHT / 2) / ball.MAX_VEL
        ball.y_vel = -difference_in_y / reduction_factor

    def _check_paddle_collision(self, paddle: Paddle, is_left: bool) -> None:
        """
        Checks if the ball collides with the given paddle. If so, reflect the ball
        and increment the appropriate hit counter.
        """
        ball = self.ball

        # If the ball is moving away from this paddle, no need to check
        if is_left and ball.x_vel >= 0:
            return
        if not is_left and ball.x_vel <= 0:
            return

        # Check if ball is vertically within paddle range
        if paddle.y <= ball.y <= paddle.y + Paddle.HEIGHT:
            if is_left:
                # Check left paddle collision
                if ball.x - ball.RADIUS <= paddle.x + Paddle.WIDTH:
                    self._reflect_ball(ball, paddle)
                    self.left_hits += 1
            else:
                # Check right paddle collision
                if ball.x + ball.RADIUS >= paddle.x:
                    self._reflect_ball(ball, paddle)
                    self.right_hits += 1

    def _handle_collision(self) -> None:
        """
        Checks and handles collisions between the ball and the game boundaries or paddles.
        """
        ball = self.ball

        # Handle collision with top and bottom walls
        if ball.y + ball.RADIUS >= self.window_height or ball.y - ball.RADIUS <= 0:
            ball.y_vel *= -1

        # Handle collision with each paddle
        self._check_paddle_collision(self.left_paddle, is_left=True)
        self._check_paddle_collision(self.right_paddle, is_left=False)

    def draw(self, draw_score: bool = True, draw_hits: bool = False) -> None:
        """
        Clears and redraws the game window.
        """
        self.window.fill(self.BLACK)
        self._draw_divider()

        if draw_score:
            self._draw_score()
        if draw_hits:
            self._draw_hits()

        self.left_paddle.draw(self.window)
        self.right_paddle.draw(self.window)
        self.ball.draw(self.window)

    def move_paddle(self, left: bool = True, up: bool = True) -> bool:
        """
        Moves the specified paddle up or down, ensuring it remains within the screen.
        """
        if left:
            if up and self.left_paddle.y - Paddle.VEL < 0:
                return False
            if not up and self.left_paddle.y + Paddle.HEIGHT + Paddle.VEL > self.window_height:
                return False
            self.left_paddle.move(up)
        else:
            if up and self.right_paddle.y - Paddle.VEL < 0:
                return False
            if not up and self.right_paddle.y + Paddle.HEIGHT + Paddle.VEL > self.window_height:
                return False
            self.right_paddle.move(up)
        return True

    def loop(self) -> GameInformation:
        """
        Executes a single iteration of the game loop: moves the ball, handles collisions,
        and updates scores.
        """
        self.ball.move()
        self._handle_collision()

        # Check for scoring when the ball goes off-screen
        if self.ball.x < 0:
            self.ball.reset()
            self.right_score += 1
        elif self.ball.x > self.window_width:
            self.ball.reset()
            self.left_score += 1

        return GameInformation(
            left_hits=self.left_hits,
            right_hits=self.right_hits,
            left_score=self.left_score,
            right_score=self.right_score
        )

    def reset(self) -> None:
        """
        Resets the game state, including the ball, paddles, scores, and hit counts.
        """
        self.ball.reset()
        self.left_paddle.reset()
        self.right_paddle.reset()
        self.left_score = 0
        self.right_score = 0
        self.left_hits = 0
        self.right_hits = 0
