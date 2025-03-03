# game_env.py

import numpy as np
import pygame
from dino_game import Dino, Obstacle, Bird, Boulder, PowerUp, GROUND_LEVEL, SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK


class DinoGameEnv:
    """
    Environment wrapper for the Dino game that follows a gym-like interface
    for reinforcement learning.
    """

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.game_speed = 7  # Base game speed
        self.action_space = 3  # 0: do nothing, 1: jump, 2: duck
        self.observation_space_shape = (8,)  # State features

        # Initialize instance attributes to fix "defined outside __init__" warnings
        self.dino = None
        self.obstacles = []
        self.powerups = []
        self.score = 0
        self.game_over = False
        self.spawn_timer = 0
        self.powerup_timer = 0
        self.clouds = []

        # Initialize pygame if rendering is enabled
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Dino Game - Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)

        self.reset()

    def reset(self):
        """Reset the environment and return initial observation."""
        self.dino = Dino()
        self.obstacles = []
        self.powerups = []
        self.score = 0
        self.game_over = False
        self.spawn_timer = 0
        self.powerup_timer = 0

        # Initialize clouds for background if rendering
        if self.render_mode == 'human' and not self.clouds:
            self.clouds = [self._create_cloud(cloud_x) for cloud_x in np.random.randint(0, SCREEN_WIDTH, size=3)]

        # Handle pygame events to prevent freezing
        if self.render_mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        # Return initial observation
        return self._get_observation()

    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action (int): 0 = do nothing, 1 = jump, 2 = duck

        Returns:
            tuple: (observation, reward, done, info)
        """
        if self.game_over:
            return self._get_observation(), 0, True, {"score": self.score}

        # Handle pygame events to prevent freezing
        if self.render_mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        # Execute action
        if action == 1:  # Jump
            if not self.dino.jumping:
                self.dino.jump()
                self.dino.ducking = False
        elif action == 2:  # Duck
            if not self.dino.jumping:
                self.dino.ducking = True
        else:  # Do nothing (stop ducking if already ducking)
            if self.dino.ducking:
                self.dino.ducking = False

        # Update game state
        self.dino.update()

        # Update clouds if rendering
        if self.render_mode == 'human':
            for cloud in self.clouds:
                cloud.update()
            # Remove off-screen clouds
            self.clouds = [c for c in self.clouds if c.x + c.width > 0]
            # Ensure there are always ~3 clouds
            while len(self.clouds) < 3:
                self.clouds.append(self._create_cloud(SCREEN_WIDTH + np.random.randint(0, 100)))

        # Spawn obstacles
        self.spawn_timer += 1
        if self.spawn_timer > np.random.randint(60, 120):
            r = np.random.random()
            if r < 0.3:
                self.obstacles.append(Bird(SCREEN_WIDTH))
            elif r < 0.8:
                self.obstacles.append(Obstacle(SCREEN_WIDTH))
            else:
                self.obstacles.append(Boulder(SCREEN_WIDTH))
            self.spawn_timer = 0

        # Spawn power-ups
        self.powerup_timer += 1
        if self.powerup_timer > np.random.randint(300, 600):
            self.powerups.append(PowerUp(SCREEN_WIDTH))
            self.powerup_timer = 0

        # Update obstacles and power-ups
        for current_obstacle in self.obstacles:
            current_obstacle.update(self.game_speed)
        for current_powerup in self.powerups:
            current_powerup.update(self.game_speed)

        # Remove off-screen objects
        self.obstacles = self._remove_offscreen_items(self.obstacles)
        self.powerups = self._remove_offscreen_items(self.powerups)

        # Check for power-up collection
        dino_rect = self.dino.get_rect()
        for current_powerup in self.powerups[:]:
            if dino_rect.colliderect(current_powerup.get_rect()):
                if current_powerup.type == "shield":
                    self.dino.shield_timer = 60 * 2  # 2 seconds at 60 FPS
                self.powerups.remove(current_powerup)

        # Check for collisions with obstacles
        if self.dino.shield_timer <= 0:
            for current_obstacle in self.obstacles:
                if dino_rect.colliderect(current_obstacle.get_rect()):
                    self.game_over = True
                    return self._get_observation(), -100, True, {"score": self.score}

        # Calculate reward (living + score)
        reward = 0.1  # Small reward for surviving
        # Give more reward when successfully navigating past obstacles
        for current_obstacle in self.obstacles:
            if current_obstacle.x + (current_obstacle.width if hasattr(current_obstacle,
                                                                       'width') else current_obstacle.radius * 2) < self.dino.x and not hasattr(
                    current_obstacle, 'passed'):
                current_obstacle.passed = True
                reward += 1.0

        self.score += 1

        # Render if needed
        if self.render_mode == 'human':
            self._render_frame()

        return self._get_observation(), reward, self.game_over, {"score": self.score}

    def render(self):
        """Render the current game state to the screen."""
        if self.render_mode == 'human':
            self._render_frame()

    def _render_frame(self):
        """Render the game state to the screen."""
        if self.render_mode != 'human':
            return

        self.screen.fill(WHITE)

        # Draw clouds (background)
        for cloud in self.clouds:
            cloud.draw(self.screen)

        # Ground line
        pygame.draw.line(self.screen, BLACK, (0, GROUND_LEVEL), (SCREEN_WIDTH, GROUND_LEVEL), 2)

        # Draw dino, obstacles, and powerups
        self.dino.draw(self.screen)
        for current_obstacle in self.obstacles:
            current_obstacle.draw(self.screen)
        for current_powerup in self.powerups:
            current_powerup.draw(self.screen)

        # Display score
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)  # Maintain 60 FPS

    def _get_observation(self):
        """
        Extract relevant features from the game state as the observation.

        Returns:
            ndarray: Observation vector with game features
        """
        # Find the closest obstacle and power-up
        closest_obstacle_dist = SCREEN_WIDTH
        closest_obstacle_height = 0
        closest_obstacle_width = 0
        closest_obstacle_y = 0

        if self.obstacles:
            # Sort obstacles by x position (closest first)
            sorted_obstacles = sorted(self.obstacles, key=lambda sorted_obs: sorted_obs.x)
            for current_obstacle in sorted_obstacles:
                # Only consider obstacles that are ahead of the dino
                if current_obstacle.x >= self.dino.x:
                    closest_obstacle_dist = current_obstacle.x - self.dino.x
                    if hasattr(current_obstacle, 'height'):
                        closest_obstacle_height = current_obstacle.height
                        closest_obstacle_width = current_obstacle.width
                        closest_obstacle_y = current_obstacle.y
                    else:  # Boulder
                        closest_obstacle_height = current_obstacle.radius * 2
                        closest_obstacle_width = current_obstacle.radius * 2
                        closest_obstacle_y = current_obstacle.y - current_obstacle.radius
                    break

        # Create observation vector
        obs = np.array([
            closest_obstacle_dist / SCREEN_WIDTH,  # Distance to the closest obstacle
            closest_obstacle_height / SCREEN_HEIGHT,  # Height of closest obstacle
            closest_obstacle_width / SCREEN_WIDTH,  # Width of closest obstacle
            closest_obstacle_y / GROUND_LEVEL,  # Y position of closest obstacle
            1.0 if self.dino.jumping else 0.0,  # Is dino jumping?
            1.0 if self.dino.ducking else 0.0,  # Is dino ducking?
            self.dino.vel_y / 15.0,  # Dino's vertical velocity (normalized)
            self.dino.shield_timer / (60 * 2)  # Shield timer (normalized)
        ])

        return obs

    @staticmethod
    def _remove_offscreen_items(items):
        """Remove items that have moved off-screen."""
        filtered = []
        for current_item in items:
            if hasattr(current_item, 'width'):
                # Rectangle-based object
                if current_item.x + current_item.width > 0:
                    filtered.append(current_item)
            elif hasattr(current_item, 'radius'):
                # Boulder
                if current_item.x + current_item.radius > 0:
                    filtered.append(current_item)
        return filtered

    @staticmethod
    def _create_cloud(cloud_x):
        """Create a cloud object for the background."""

        class Cloud:
            def __init__(self, x, y, width, height, speed):
                self.x = x
                self.y = y
                self.width = width
                self.height = height
                self.speed = speed

            def update(self):
                self.x -= self.speed

            def draw(self, surface):
                pygame.draw.ellipse(surface, (200, 200, 200),
                                    (self.x, self.y, self.width, self.height))

        return Cloud(
            x=cloud_x,
            y=np.random.randint(20, 100),
            width=np.random.randint(60, 100),
            height=np.random.randint(30, 50),
            speed=np.random.uniform(1, 2)
        )