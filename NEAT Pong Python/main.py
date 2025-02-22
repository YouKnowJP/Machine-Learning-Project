from pong import Game
import pygame
import neat
import os
import time
import pickle

# -------------------- Constants -------------------- #
FPS = 720
WIDTH, HEIGHT = 700, 500
MAX_HITS = 50
CHECKPOINT_FILE = "neat-checkpoint-2"
BEST_MODEL_FILE = "best.pickle"

# -------------------- PongGame Class -------------------- #
class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.ball = self.game.ball
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle

        # Initialize genome attributes to avoid warnings
        self.genome1 = None
        self.genome2 = None

    def test_ai(self, net):
        """ Test the AI against a human player using the trained NEAT model. """
        clock = pygame.time.Clock()
        run = True

        while run:
            clock.tick(FPS)

            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            # AI movement decision
            output = net.activate((self.right_paddle.y, abs(self.right_paddle.x - self.ball.x), self.ball.y))
            decision = output.index(max(output))
            if decision == 1:
                self.game.move_paddle(left=False, up=True)  # Move up
            elif decision == 2:
                self.game.move_paddle(left=False, up=False)  # Move down

            # Human controls for the left paddle
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            elif keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            self.game.draw(draw_score=True)
            pygame.display.update()

    def train_ai(self, genome1, genome2, config, draw=False):
        """ Train two AI agents and update their fitness. """
        start_time = time.time()
        self.genome1, self.genome2 = genome1, genome2

        # Create neural networks for each genome
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        while True:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

            game_info = self.game.loop()
            self.move_ai_paddles(net1, net2)

            if draw:
                self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()

            total_hits = game_info.left_hits + game_info.right_hits
            duration = time.time() - start_time

            # End game if a score is reached or total hits reach/exceed MAX_HITS
            if game_info.left_score == 1 or game_info.right_score == 1 or total_hits >= MAX_HITS:
                self.calculate_fitness(game_info, duration)
                break

        return False

    def move_ai_paddles(self, net1, net2):
        """ Move AI paddles based on neural networks' decisions. """
        for genome, net, paddle, is_left in [
            (self.genome1, net1, self.left_paddle, True),
            (self.genome2, net2, self.right_paddle, False)
        ]:
            output = net.activate((paddle.y, abs(paddle.x - self.ball.x), self.ball.y))
            decision = output.index(max(output))

            if decision == 0:
                genome.fitness = (genome.fitness or 0) - 0.01  # Discourage inaction
            elif decision == 1:
                valid = self.game.move_paddle(left=is_left, up=True)  # Move up
                if not valid:
                    genome.fitness -= 1
            else:
                valid = self.game.move_paddle(left=is_left, up=False)  # Move down
                if not valid:
                    genome.fitness -= 1

    def calculate_fitness(self, game_info, duration):
        """ Update fitness based on hits and duration. """
        self.genome1.fitness = (self.genome1.fitness or 0) + game_info.left_hits + duration
        self.genome2.fitness = (self.genome2.fitness or 0) + game_info.right_hits + duration

# -------------------- NEAT Evaluation Functions -------------------- #
def eval_genomes(genomes, config):
    """ Evaluate genomes by running Pong games. """
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong AI Training")

    for i, (genome_id1, genome1) in enumerate(genomes):
        genome1.fitness = genome1.fitness if genome1.fitness is not None else 0
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = genome2.fitness if genome2.fitness is not None else 0
            pong = PongGame(win, WIDTH, HEIGHT)
            force_quit = pong.train_ai(genome1, genome2, config, draw=True)
            if force_quit:
                pygame.quit()
                return

def run_neat(config):
    """ Run NEAT training with checkpoint restoration. """
    try:
        population = neat.Checkpointer.restore_checkpoint(CHECKPOINT_FILE)
        print(f"Loaded checkpoint '{CHECKPOINT_FILE}'")
    except FileNotFoundError:
        print("No checkpoint found. Starting fresh training.")
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))

    winner = population.run(eval_genomes, 50)

    # Save the best genome safely
    with open(BEST_MODEL_FILE, "wb") as f:
        pickle.dump(winner, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Best model saved to '{BEST_MODEL_FILE}'")

def test_best_network(config):
    """ Load and test the best trained network. """
    if not os.path.exists(BEST_MODEL_FILE):
        print("Error: best.pickle not found! Train the model first.")
        return

    with open(BEST_MODEL_FILE, "rb") as f:
        winner = pickle.load(f)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong AI - Best Model")
    pong = PongGame(win, WIDTH, HEIGHT)
    pong.test_ai(winner_net)

# -------------------- Main -------------------- #
def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    run_neat(config)
    test_best_network(config)

if __name__ == '__main__':
    main()
