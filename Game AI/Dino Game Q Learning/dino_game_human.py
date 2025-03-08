# dino_game_human.py

import pygame
import random
import sys

# ---------------------------
#       PYGAME SETUP
# ---------------------------
pygame.init()

# ---------------------------
#       SCREEN SETTINGS
# ---------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
GROUND_LEVEL = 300
FPS = 60

# ---------------------------
#       SPAWN CONSTANTS
# ---------------------------
OBSTACLE_SPAWN_MIN = 60
OBSTACLE_SPAWN_MAX = 120
POWERUP_SPAWN_MIN  = 300
POWERUP_SPAWN_MAX  = 600

# ---------------------------
#       COLORS
# ---------------------------
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GREEN  = (0,   200, 0)
RED    = (200, 0,   0)
BLUE   = (0,   0,   255)
YELLOW = (255, 255, 0)
GRAY   = (128, 128, 128)

# ---------------------------
#       PYGAME OBJECTS
# ---------------------------
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Dino Game")
clock = pygame.time.Clock()


# ---------------------------
#       CLASSES
# ---------------------------
class Dino:
    def __init__(self):
        self.width = 50
        self.standing_height = 50
        self.duck_height = 30
        self.height = self.standing_height
        self.x = 50
        self.y = GROUND_LEVEL - self.height
        self.vel_y = 0
        self.jumping = False
        self.ducking = False
        self.gravity = 0.8
        self.shield_timer = 0  # Shield power-up duration (in frames)

    def jump(self):
        """Trigger a jump if not already in the air."""
        if not self.jumping:
            self.jumping = True
            self.ducking = False  # Cancel ducking when jumping
            self.height = self.standing_height
            self.vel_y = -15

    def update(self):
        """Update Dino's position and shield timer."""
        if self.shield_timer > 0:
            self.shield_timer -= 1

        if self.jumping:
            self.y += self.vel_y
            self.vel_y += self.gravity
            if self.y >= GROUND_LEVEL - self.height:
                self.y = GROUND_LEVEL - self.height
                self.jumping = False
                self.vel_y = 0
        else:
            # Adjust height based on ducking
            if self.ducking:
                self.height = self.duck_height
            else:
                self.height = self.standing_height
            self.y = GROUND_LEVEL - self.height

    def draw(self, surface):
        """Draw the Dino (with shield glow if active)."""
        if self.shield_timer > 0:
            # Draw a yellow box around the dino to indicate shield
            pygame.draw.rect(surface, YELLOW,
                             (self.x - 5, self.y - 5,
                              self.width + 10, self.height + 10))
        pygame.draw.rect(surface, GREEN, (self.x, self.y, self.width, self.height))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Obstacle:
    """Tree obstacle with variable width."""
    def __init__(self, x):
        multiplier = random.randint(1, 3)
        base_width = 20
        self.width = base_width * multiplier
        self.height = 50
        self.x = x
        self.y = GROUND_LEVEL - self.height

    def update(self, game_speed):
        self.x -= game_speed

    def draw(self, surface):
        pygame.draw.rect(surface, RED, (self.x, self.y, self.width, self.height))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Bird:
    """Bird obstacle with variable speed offset and two flight heights."""
    def __init__(self, x):
        self.width = 50
        self.height = 40
        self.x = x
        # Random offset to make some birds faster/slower
        self.speed_offset = random.choice([0, 2, 4])
        # Bird can either require jump or duck
        self.condition = random.choice(["jump", "duck"])
        if self.condition == "jump":
            self.y = GROUND_LEVEL - self.height - 5  # Low bird
        else:
            self.y = 215  # High bird

    def update(self, game_speed):
        self.x -= (game_speed + self.speed_offset)

    def draw(self, surface):
        pygame.draw.rect(surface, BLUE, (self.x, self.y, self.width, self.height))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Boulder:
    """Rolling boulder obstacle."""
    def __init__(self, x):
        self.radius = 20
        self.x = x
        self.y = GROUND_LEVEL - self.radius

    def update(self, game_speed):
        self.x -= game_speed

    def draw(self, surface):
        pygame.draw.circle(surface, GRAY, (int(self.x), int(self.y)), self.radius)

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius,
                           self.radius * 2, self.radius * 2)


class PowerUp:
    """Shield power-up spawns above ground; Dino can collect for brief invincibility."""
    def __init__(self, x):
        self.width = 30
        self.height = 30
        self.x = x
        self.y = GROUND_LEVEL - self.height - 60
        self.type = "shield"

    def update(self, game_speed):
        self.x -= game_speed

    def draw(self, surface):
        pygame.draw.rect(surface, YELLOW, (self.x, self.y, self.width, self.height))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Cloud:
    """Parallax cloud that moves slower than obstacles for background effect."""
    def __init__(self, x):
        self.x = x
        self.y = random.randint(20, 100)
        self.width = random.randint(60, 100)
        self.height = random.randint(30, 50)
        self.speed = random.uniform(1, 2)

    def update(self):
        self.x -= self.speed

    def draw(self, surface):
        pygame.draw.ellipse(surface, (200, 200, 200),
                            (self.x, self.y, self.width, self.height))


# ---------------------------
#       HELPER FUNCTIONS
# ---------------------------
def remove_offscreen_items(items):
    """
    Remove items (obstacles/powerups) that have completely moved off-screen.
    Works for both rectangular and circular items (boulders).
    """
    filtered = []
    for obj in items:
        if hasattr(obj, 'width'):
            # If it's a rectangle-based object
            if obj.x + obj.width > 0:
                filtered.append(obj)
        elif hasattr(obj, 'radius'):
            # If it's a boulder
            if obj.x + obj.radius > 0:
                filtered.append(obj)
    return filtered


# ---------------------------
#       MAIN GAME LOOP
# ---------------------------
def run_game():
    dino = Dino()
    obstacles = []
    powerups = []
    clouds = [Cloud(random.randint(0, SCREEN_WIDTH)) for _ in range(3)]
    spawn_timer = 0
    powerup_timer = 0
    score = 0
    game_over = False
    font = pygame.font.SysFont(None, 36)

    base_game_speed = 7  # Constant speed for obstacles

    while True:
        clock.tick(FPS)
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if game_over:
                        return  # Restart game loop
                    else:
                        dino.jump()
                elif event.key == pygame.K_DOWN:
                    if not dino.jumping:
                        dino.ducking = True

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    dino.ducking = False

        if not game_over:
            # Update Dino
            dino.update()

            # Update clouds (parallax background)
            for cloud in clouds:
                cloud.update()
            # Remove offscreen clouds
            clouds = [c for c in clouds if c.x + c.width > 0]
            # Ensure there are always ~3 clouds
            while len(clouds) < 3:
                clouds.append(Cloud(SCREEN_WIDTH + random.randint(0, 100)))

            # Spawn obstacles (tree, bird, boulder) at random intervals
            spawn_timer += 1
            if spawn_timer > random.randint(OBSTACLE_SPAWN_MIN, OBSTACLE_SPAWN_MAX):
                r = random.random()
                if r < 0.3:
                    obstacles.append(Bird(SCREEN_WIDTH))
                elif r < 0.8:
                    obstacles.append(Obstacle(SCREEN_WIDTH))
                else:
                    obstacles.append(Boulder(SCREEN_WIDTH))
                spawn_timer = 0

            # Spawn power-ups occasionally
            powerup_timer += 1
            if powerup_timer > random.randint(POWERUP_SPAWN_MIN, POWERUP_SPAWN_MAX):
                powerups.append(PowerUp(SCREEN_WIDTH))
                powerup_timer = 0

            # Update obstacles & power-ups
            for obs in obstacles:
                obs.update(base_game_speed)
            for pu in powerups:
                pu.update(base_game_speed)

            # Remove off-screen obstacles & power-ups
            obstacles = remove_offscreen_items(obstacles)
            powerups = remove_offscreen_items(powerups)

            # Check collision with power-ups
            dino_rect = dino.get_rect()
            for pu in powerups[:]:  # Iterate over a copy for safe removal
                if dino_rect.colliderect(pu.get_rect()):
                    if pu.type == "shield":
                        # Activate shield for 5 seconds
                        dino.shield_timer = FPS * 5
                    powerups.remove(pu)

            # Collision detection with obstacles (ignored if shield is active)
            if dino.shield_timer <= 0:
                for obs in obstacles:
                    if dino_rect.colliderect(obs.get_rect()):
                        game_over = True
                        break

            score += 1

        # ---------------------------
        #       DRAW SECTION
        # ---------------------------
        screen.fill(WHITE)

        # Draw clouds (background)
        for cloud in clouds:
            cloud.draw(screen)

        # Ground line
        pygame.draw.line(screen, BLACK, (0, GROUND_LEVEL),
                         (SCREEN_WIDTH, GROUND_LEVEL), 2)

        # Dino, obstacles, power-ups
        dino.draw(screen)
        for obs in obstacles:
            obs.draw(screen)
        for pu in powerups:
            pu.draw(screen)

        # Score display
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))

        # Game Over message
        if game_over:
            over_text = font.render("Game Over! Press Space to Restart", True, BLACK)
            screen.blit(over_text, (SCREEN_WIDTH // 2 - over_text.get_width() // 2,
                                    SCREEN_HEIGHT // 2))

        pygame.display.flip()


def main():
    while True:
        run_game()


if __name__ == "__main__":
    main()
