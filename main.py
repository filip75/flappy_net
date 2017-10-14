import sys, pygame, random
import numpy as np
import pickle
from network import *

# window
WIDTH = 640
HEIGHT = 480
FONT_SIZE = 16

# game settings
H_SPEED = 220 / 1000  # pixels per milisecond
GRAVITY = 2200 / 1000000  # acceleration pixels per milisecond squared
JUMP_SPEED = -650 / 1000  # speed after jump pixels per milisecond squared
FPS = 60
INTERVAL = (1000 / FPS)  # time between frames [miliseconds]
JUMP_DELAY = 200  # [miliseconds]
BIRDS_NUMBER = 32

# learning settings
ROUNDS_PER_GENERATION = 5
GENERATIONS = 200
MAX_DISTANCE = 50000

PILLAR_WIDTH = 60
PILLAR_H_GAP = 240  # distance between pillars
PILLAR_V_GAP = int(0.3 * HEIGHT)
GAP_RANGE = int(0.3 * HEIGHT)  # max distance between centre of gap and centre of screen
FIRST_PILLAR_DISTANCE = 800  # distance to centre of first pillar
PILLARS_NUMBER = 3  # number of pillars created ahead

BIRD_X_POS = 50  # pixels from left edge
BIRD_START_Y_POS = HEIGHT / 2
BIRD_RADIUS = 15
BIRD_RADIUS_SQUARED = BIRD_RADIUS * BIRD_RADIUS

# colours
WHITE = 0xff, 0xff, 0xff
BLACK = 0, 0, 0

FONTS = pygame.image.load("font.png")
PILLAR_IMAGE = pygame.image.load("pillar.bmp")
BIRD_IMAGE = pygame.image.load("bird.png")
BIRD_IMAGE = pygame.transform.scale(BIRD_IMAGE, (2 * BIRD_RADIUS, 2 * BIRD_RADIUS))


def display_string(surface, x, y, string):
    x_c = x
    y_c = y
    for i in range(0, len(string)):
        ascii_code = ord(string[i])
        if ascii_code != ord('\n'):
            font_y = int(ascii_code / 16)
            font_x = ascii_code % 16
            rect = (font_x * FONT_SIZE, font_y * FONT_SIZE, FONT_SIZE, FONT_SIZE)
            char = FONTS.subsurface(rect)
            dest_rect = (x_c, y_c, FONT_SIZE, FONT_SIZE)
            surface.blit(char, dest_rect)
        if (ascii_code == ord('\n')):
            x_c = x
            y_c += FONT_SIZE
        else:
            x_c += FONT_SIZE


class Pillar:
    def __init__(self, x_position, image=None):
        self.gap_position = (HEIGHT / 2) + random.randint(-GAP_RANGE, GAP_RANGE)
        top_height = int(self.gap_position - PILLAR_V_GAP / 2)
        bottom_height = int(HEIGHT - self.gap_position + (PILLAR_V_GAP / 2))

        if image is not None:
            self.top = pygame.transform.scale(image, (PILLAR_WIDTH, top_height))
            self.top_rect = self.top.get_rect()
            self.bottom = pygame.transform.scale(image, (PILLAR_WIDTH, bottom_height))
            self.bottom_rect = self.bottom.get_rect()
            self.bottom_rect[1] = self.gap_position + (PILLAR_V_GAP / 2)
        else:
            self.top_rect = pygame.Rect(0, 0, PILLAR_WIDTH, top_height)
            self.bottom_rect = pygame.Rect(0, self.gap_position + (PILLAR_V_GAP / 2), PILLAR_WIDTH, bottom_height)

        self.start_position = x_position
        self.time = 0
        self.move(0)

    def move(self, time):
        self.time = self.time + time
        self.top_rect[0] = int(self.start_position - H_SPEED * self.time - PILLAR_WIDTH / 2)
        self.bottom_rect[0] = int(self.start_position - H_SPEED * self.time - PILLAR_WIDTH / 2)

    def display(self, surface):
        surface.blit(self.top, self.top_rect)
        surface.blit(self.bottom, self.bottom_rect)

    def get_position(self):
        # self.top_rect[0] is position of left edge, this returns position of centre of pillar
        return [self.top_rect[0] + PILLAR_WIDTH / 2, self.gap_position]

    def is_collision(self, bird):
        t_x = self.top_rect[0] + PILLAR_WIDTH / 2
        t_y = self.top_rect[1] + self.top_rect[3]
        b_x = self.bottom_rect[0] + PILLAR_WIDTH / 2  # = t_x
        b_y = self.bottom_rect[1]

        tl_corner = (bird.x_pos - (t_x - PILLAR_WIDTH / 2)) ** 2 + (bird.y_pos - t_y) ** 2
        bl_corner = (bird.x_pos - (b_x - PILLAR_WIDTH / 2)) ** 2 + (bird.y_pos - b_y) ** 2
        tr_corner = (bird.x_pos - (t_x + PILLAR_WIDTH / 2)) ** 2 + (bird.y_pos - t_y) ** 2
        br_corner = (bird.x_pos - (b_x + PILLAR_WIDTH / 2)) ** 2 + (bird.y_pos - b_y) ** 2

        if tl_corner < BIRD_RADIUS_SQUARED or bl_corner < BIRD_RADIUS_SQUARED or tr_corner < BIRD_RADIUS_SQUARED or br_corner < BIRD_RADIUS_SQUARED:
            return True

        if abs(bird.x_pos - t_x) < BIRD_RADIUS + PILLAR_WIDTH / 2:
            if bird.y_pos < t_y or bird.y_pos > b_y:
                return True
        if abs(bird.x_pos - t_x) < PILLAR_WIDTH / 2:
            if bird.y_pos - BIRD_RADIUS < t_y or bird.y_pos + BIRD_RADIUS > b_y:
                return True
        return False


class Bird:
    def __init__(self):
        self.y_pos = BIRD_START_Y_POS
        self.x_pos = BIRD_X_POS
        self.velocity = 0
        self.rect = pygame.Rect(self.x_pos - BIRD_RADIUS, self.y_pos - BIRD_RADIUS, 2 * BIRD_RADIUS, 2 * BIRD_RADIUS)
        self.fitness = 0
        self.is_alive = True
        self.jump_delay = 0

    def reset(self):
        self.y_pos = BIRD_START_Y_POS
        self.x_pos = BIRD_X_POS
        self.velocity = 0
        self.rect = pygame.Rect(self.x_pos - BIRD_RADIUS, self.y_pos - BIRD_RADIUS, 2 * BIRD_RADIUS, 2 * BIRD_RADIUS)
        self.is_alive = True

    def move(self, time_elapsed):
        self.jump_delay -= time_elapsed
        self.y_pos += self.velocity * time_elapsed + (GRAVITY * time_elapsed ** 2) / 2
        self.velocity += GRAVITY * time_elapsed
        self.rect[1] = self.y_pos - BIRD_RADIUS
        self.fitness += time_elapsed

    def jump(self):
        if self.jump_delay <= 0:
            self.velocity = JUMP_SPEED
            self.jump_delay = JUMP_DELAY

    def display(self, surface, image):
        surface.blit(image, self.rect)


class Game:
    def __init__(self, birds_number, pillar_image=None, bird_image=None):
        self.pillars = []
        self.pillar_image = pillar_image
        for i in range(0, PILLARS_NUMBER):
            self.pillars.append(Pillar(FIRST_PILLAR_DISTANCE + i * PILLAR_H_GAP, self.pillar_image))

        self.birds = [Bird() for _ in range(birds_number)]
        self.bird_image = bird_image

    def reset(self):
        self.pillars.clear()
        for i in range(0, PILLARS_NUMBER):
            self.pillars.append(Pillar(FIRST_PILLAR_DISTANCE + i * PILLAR_H_GAP, self.pillar_image))
        for bird in self.birds:
            bird.reset()

    def update(self, time_elapsed):
        for pillar in self.pillars:
            pillar.move(time_elapsed)

        while self.pillars[0].get_position()[0] < -PILLAR_WIDTH / 2:
            # bird passed the pillar
            self.pillars.pop(0)
            self.pillars.append(
                Pillar(self.pillars[len(self.pillars) - 1].get_position()[0] + PILLAR_H_GAP, self.pillar_image))

        for bird in self.birds:
            if bird.is_alive:
                bird.move(time_elapsed)
                if self.pillars[0].is_collision(
                        bird) or bird.y_pos + BIRD_RADIUS > HEIGHT or bird.y_pos - BIRD_RADIUS < 0:
                    bird.is_alive = False

    def jump(self, bird_number):
        self.birds[bird_number].jump()

    def get_birds_info(self):
        # [is_alive, horizontal_distance_to_next_pillar, verical_distance_to_centre_of_next_gap]
        info = np.zeros([len(self.birds), 3])
        for i in range(0, len(self.birds)):
            if self.birds[i].is_alive:
                info[i, 0] = 1
                if self.pillars[0].get_position()[0] + PILLAR_WIDTH / 2 > self.birds[i].x_pos:
                    info[i, 1] = (self.pillars[0].get_position()[0] + PILLAR_WIDTH / 2 - self.birds[
                        i].x_pos) / PILLAR_H_GAP
                    info[i, 2] = (self.pillars[0].get_position()[1] - self.birds[i].y_pos) / PILLAR_H_GAP
                else:
                    info[i, 1] = (self.pillars[1].get_position()[0] + PILLAR_WIDTH / 2 - self.birds[
                        i].x_pos) / PILLAR_H_GAP
                    info[i, 2] = (self.pillars[1].get_position()[1] - self.birds[i].y_pos) / PILLAR_H_GAP
            else:
                info[i, 0] = 0
        return info

    def get_fitness(self):
        return [bird.fitness for bird in self.birds]

    def display(self, surface):
        for pillar in self.pillars:
            pillar.display(surface)
        for bird in self.birds:
            if bird.is_alive:
                bird.display(surface, self.bird_image)


class Player:
    def __init__(self, number_of_birds, network=None):
        self.game = Game(number_of_birds, PILLAR_IMAGE, BIRD_IMAGE)
        if network is None:
            self.network_manager = Network_manager(BIRDS_NUMBER, 2, 3, [10, 10, 1])
        else:
            file = open(network, 'rb')
            self.network = pickle.load(file)
            file.close()

    def just_play(self, output):
        generations = 0
        time_elapsed = 0
        games = 0
        max_fitness = 0
        while generations < GENERATIONS:
            time_elapsed += INTERVAL
            self.game.update(INTERVAL)
            info = self.game.get_birds_info()
            steering = self.network_manager.evaluate(info)
            for i in range(0, BIRDS_NUMBER):
                if steering[i] > 0.5:
                    self.game.jump(i)

            if np.sum(info, 0)[0] == 0 or time_elapsed > MAX_DISTANCE:
                # all birds are dead or reached max distance
                if games == ROUNDS_PER_GENERATION:
                    games = 0
                    generations += 1
                    print(str(int(np.amax(self.game.get_fitness()) / 1000)), self.game.get_fitness())
                    if np.amax(self.game.get_fitness()) > max_fitness:
                        file = open(output, 'wb')
                        pickle.dump(self.network_manager.networks[np.argmax(self.game.get_fitness())], file)
                        file.close()
                    self.network_manager.new_generation(self.game.get_fitness())
                    self.game = Game(BIRDS_NUMBER, PILLAR_IMAGE, BIRD_IMAGE)
                else:
                    games += 1
                    time_elapsed = 0
                    self.game.reset()

    def play_and_watch(self):
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()
        games = 0
        time = 0
        generation = 1
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            screen.fill(WHITE)
            time_elapsed = clock.tick(FPS)
            time += time_elapsed
            self.game.update(time_elapsed)
            self.game.display(screen)
            info = self.game.get_birds_info()
            steering = self.network_manager.evaluate(info)
            for i in range(0, BIRDS_NUMBER):
                if steering[i] > 0.5:
                    self.game.jump(i)

            fps = int(clock.get_fps())
            display_string(screen, WIDTH - FONT_SIZE * 8, 0 * FONT_SIZE, "FPS: " + str(fps))
            display_string(screen, WIDTH - FONT_SIZE * 9, 1 * FONT_SIZE, "time: " + str(int(time / 1000)))
            display_string(screen, WIDTH - FONT_SIZE * 8, 2 * FONT_SIZE, "gen: " + str(int(generation)))
            pygame.display.flip()
            if np.sum(info, 0)[0] == 0:
                if games == ROUNDS_PER_GENERATION:
                    games = 0
                    print(np.amax(self.game.get_fitness()), self.game.get_fitness())
                    self.network_manager.new_generation(self.game.get_fitness())
                    self.game = Game(BIRDS_NUMBER, PILLAR_IMAGE, BIRD_IMAGE)
                    time = 0
                    generation += 1
                else:
                    games += 1
                    self.game.reset()
                    time = 0

    def play_trained(self):
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()
        time = 0
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            screen.fill(WHITE)
            time_elapsed = clock.tick(FPS)
            time += time_elapsed
            self.game.update(time_elapsed)
            self.game.display(screen)
            info = self.game.get_birds_info()
            steering = self.network.evaluate(info[:, 1:])
            if steering > 0.5:
                self.game.jump(0)

            fps = int(clock.get_fps())
            display_string(screen, WIDTH - FONT_SIZE * 8, 0 * FONT_SIZE, "FPS: " + str(fps))
            display_string(screen, WIDTH - FONT_SIZE * 9, 1 * FONT_SIZE, "time: " + str(int(time / 1000)))
            pygame.display.flip()
            if np.sum(info, 0)[0] == 0:
                self.game.reset()
                time = 0


# player = Player(BIRDS_NUMBER)
# player.just_play('net.n')

player = Player(1, 'best.n')
player.play_trained()
