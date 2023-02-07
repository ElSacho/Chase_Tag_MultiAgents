from plateau import Plateau
import pygame
from utils import colors, size
import numpy as np
import math
from cat import Cat
import random


class CatState(Cat):
    def __init__(self, starting_position, plateau, vision = 2):
        super().__init__(starting_position, plateau)
        self.vision = vision
        # La vision plus l'ecart de position avec le chasseur
        self.observation_space = (2*vision+1)**2+2
        self.action_space = 5
            
    # Get current state of the game  
    def get_state(self, mouse):
        self.view = []
        pos = int(self.pos[1]/size.BLOCK_SIZE), int(self.pos[0]/size.BLOCK_SIZE)
        for x in range(pos[0]-self.vision, pos[0]+self.vision+1):
            for y in range(pos[1]-self.vision, pos[1]+self.vision+1):
                if self.pos_isValid([x,y]):
                    case_number = x*self.plateau.n_cols + y
                    case_number_mouse = mouse.case_number
                    if case_number == case_number_mouse:
                        self.view.append(10)
                    else :
                        self.view.append(self.plateau.cases[case_number].timeToSpend)
                else :
                    self.view.append(-1)
        pos_mouse = int(mouse.pos[1]/size.BLOCK_SIZE), int(mouse.pos[0]/size.BLOCK_SIZE)
        # print([self.view[i:i+2*self.vision+1] for i in range(0, (2*self.vision+1)**2, 2*self.vision+1)])
        self.view.append(pos[0]-pos_mouse[0])
        self.view.append(pos[1]-pos_mouse[1])
        return (self.view)
                     
    def pos_isValid(self, pos):
        if pos[0] >= self.plateau.n_cols or pos[0] < 0:
            return False
        if pos[1] >= self.plateau.n_rows or pos[1] < 0:
            return False
        return True
  
    # Check if the game is finished
    def is_done(self, mouse):
        return self.hasEaten(mouse)
    # Get the reward for the current state
    def get_reward(self, mouse):
        pos = int(self.pos[1]/size.BLOCK_SIZE), int(self.pos[0]/size.BLOCK_SIZE)
        pos_mouse = int(mouse.pos[1]/size.BLOCK_SIZE), int(mouse.pos[0]/size.BLOCK_SIZE)
        distance = (pos[0]+pos_mouse[0])**2+(pos[1]+pos_mouse[1])**2
        if distance <= self.vision:
            reward = 0
        if distance == 1:
            reward = 0
        if distance == 0:
            reward = 100/self.step
        if distance > self.vision :
            reward = 0
        return reward

    # Get all possible actions for the current state
    def get_actions(self):
        return [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]

    # Take an action and return the next state of the game
    def take_action(self , number_action, mouse):
        # action = [0] * 5
        # action[random.randint(0, 4)] = 1
        action = self.get_actions()[number_action]
        self.move(action)
        return self.get_state(mouse)

class Cats():
    def __init__(self, starting_position, plateau, number_of_cats, vision = 2):
        self.number_of_cats = number_of_cats
        self.tabCats = []
        for _ in range(number_of_cats):
            self.tabCats.append(CatState(starting_position, plateau, vision))
        self.observation_space = self.tabCats[0].observation_space
        self.action_space = self.tabCats[0].action_space
        
    def get_state(self, mouse):
        state = []
        for cat in self.tabCats:
            state.append(cat.get_state(mouse))
        return state
        
    def draw_cats(self, screen):
        for cat in self.tabCats:
            cat.draw_cat(screen)