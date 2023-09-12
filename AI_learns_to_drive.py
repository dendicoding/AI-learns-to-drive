import pygame
import neat
import os
import time
import random
pygame.font.init()

WIN_WIDTH = 800
WIN_HEIGHT = 500

GEN = 0

CAR_IMG = pygame.image.load(os.path.join("imgs", "car.png"))
STONE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "obs.png")))
BG_IMG = pygame.image.load(os.path.join("imgs", "bg.png"))
BASE_IMG = pygame.image.load(os.path.join("imgs", "base.png"))
STAT_FONT = pygame.font.SysFont("gill sans mt", 50)

class Car:
    IMG = CAR_IMG
    MAX_ROTATION = 0
    ROT_VEL = 20
    ANIMATION_TIME = 5
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMG
        self.width = self.x
        
    def steer(self):
        self.vel = -4.5
        self.tick_count = 0
        self.width = self.x
        
        
    def move(self):
        self.tick_count += 1
        
        d = self.vel*self.tick_count + 1.5*self.tick_count**2
        
        if d >= 16:
            d = 16
            
        if d < 0:
            d -= 2
            
        self.x = self.x + d      #Steering to the left or right
        
        if d < 0 or self.x < self.width + 50:   
            if self.tilt < self.MAX_ROTATION:
                 self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > 90:
                self.tilt = self.ROT_VEL
                
    def draw(self, win):
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)        
            
    def get_mask(self):
        return pygame.mask.from_surface(self.img)
    
    
class Stone:
    GAP = 50
    VEL = 5
    
    def __init__(self, y):
        self.y = y
        self.width = 0
        
        self.left = 0
        self.right = 0
        self.STONE_LEFT = pygame.transform.flip(STONE_IMG, False, False)
        self.STONE_RIGHT = STONE_IMG
        
        self.passed = False
        self.set_width()
        
    def set_width(self):
        self.width = random.randrange(200, 550)
        self.left = self.width - self.STONE_LEFT.get_width()
        self.right = self.width + self.GAP
        
    def move(self):
        self.y -= self.VEL   #CAMBIA CON Y INVECE CHE X
        
    def draw(self, win):
        win.blit(self.STONE_LEFT, (self.left, self.y))
        win.blit(self.STONE_RIGHT, (self.right, self.y))
        
    def collide(self, car):
        car_mask = car.get_mask()
        left_mask = pygame.mask.from_surface(self.STONE_LEFT)
        right_mask = pygame.mask.from_surface(self.STONE_RIGHT)
        
        left_offset = (self.left - round(car.x), (car.y - self.y))
        right_offset = (self.right - round(car.x), (car.y - self.y))
    
        r_point = car_mask.overlap(right_mask, right_offset)
        l_point = car_mask.overlap(left_mask, left_offset)
        
        if l_point or r_point:
            return True
        
        return False
    
    
class Base:
    VEL = -5
    HEIGHT = BG_IMG.get_height() 
    IMG = BASE_IMG
    
    def __init__(self, x): 
        self.x = x
        self.y1 = 0
        self.y2 = self.HEIGHT 
        self.BASE_LEFT = pygame.transform.flip(BASE_IMG, True, True)
        self.BASE_RIGHT = BASE_IMG
        
    def move(self):
        self.y1 += self.VEL
        self.y2 += self.VEL 
        
        if self.y1 + self.HEIGHT< 0:        #se è passata
            self.y1 = self.y2 + self.HEIGHT  
         
        if self.y2 + self.HEIGHT < 0:
            self.y2 = self.y1 + self.HEIGHT    
       
    def draw(self, win):
        win.blit(self.IMG, (self.x, self.y1)) 
        win.blit(self.IMG, (self.x, self.y2)) 
        win.blit(self.BASE_LEFT, (0, self.y1))
        win.blit(self.BASE_LEFT, (0, self.y2))
        
    
def draw_window(win, cars, stones, base, score, gen):
    win.blit(BG_IMG, (0, 0))
    
    for stone in stones:
        stone.draw(win)
    
    score_text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_text, (120, 10))
    
    gen_text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(gen_text, (120, 40))
    
    cars_text = STAT_FONT.render("Cars: " + str(len(cars)), 1, (255, 255, 255))
    win.blit(cars_text, (120, 70))
    
    base.draw(win)
    for car in cars:
        car.draw(win)
    pygame.display.update()
    
def main(genomes, config):
    global GEN
    GEN += 1
    nets = []
    ge = []
    cars = []
    
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        cars.append(Car(340, 100))
        g.fitness = 0
        ge.append(g)
        
        
    
    base = Base(695)
    stones = [Stone(420)]
    pygame.display.set_caption('AI: Car learns to drive')
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    
    score = 0
    
    run = True
    while run:
        clock.tick(30)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
         
        stone_ind = 0
        if len(cars) > 0:
            if len(stones) > 1 and cars[0].y > stones[0].y + stones[0].STONE_LEFT.get_height():
                stone_ind = 1 
        else:
            run = False
            break
             
        for x, car in enumerate(cars):
            car.move()
            ge[x].fitness += 0.1
            
            output = nets[x].activate((car.x, abs(car.x - stones[stone_ind].width), abs(car.x - stones[stone_ind].right)))
            
            if output[0] > 0.5:
                car.steer()
         
        add_stone = False
        rem = []
        for stone in stones:
            for x, car in enumerate(cars):
                if stone.collide(car):
                    ge[x].fitness -= 1
                    cars.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                
                if not stone.passed and car.y > stone.y:
                    stone.passed = True
                    add_stone = True
                    
            if stone.y + stone.STONE_LEFT.get_height() < 0:   #if stone is off the screen
                rem.append(stone)   
             
            stone.move()
            
        if add_stone:
            score += 1
            for g in ge:
                g.fitness += 5
            stones.append(Stone(500))
        
        for r in rem:
            stones.remove(r)
           
        for x, car in enumerate(cars):     
            if car.x + car.img.get_width() >= 695 or car.x < 106:
                cars.pop(x)
                nets.pop(x)
                ge.pop(x)
        
        base.move()
        draw_window(win, cars, stones, base, score, GEN)
        
    

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(main, 50)    #Best Genome, you can save it as a file with Pickle


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)