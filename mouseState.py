from plateau import Plateau
import pygame
from utils import colors, size
import numpy as np
import math
from mouse import Mouse
import random


class MouseState(Mouse):
    def __init__(self, starting_position, plateau, vision = 1):
        super().__init__(starting_position, plateau)
        self.vision = vision
        # La vision plus l'ecart de position avec le chasseur
        self.observation_space = (2*vision+1)**2+2
        self.action_space = 5
      
    # Get current state of the game  
    def get_state(self, cat):
        self.view = []
        pos = int(self.pos[1]/size.BLOCK_SIZE), int(self.pos[0]/size.BLOCK_SIZE)
        for x in range(pos[0]-self.vision, pos[0]+self.vision+1):
            for y in range(pos[1]-self.vision, pos[1]+self.vision+1):
                if self.pos_isValid([x,y]):
                    case_number = x*self.plateau.n_cols + y
                    self.view.append(self.plateau.cases[case_number].timeToSpend)
                else :
                    self.view.append(-1)
        pos_cat = int(cat.pos[1]/size.BLOCK_SIZE), int(cat.pos[0]/size.BLOCK_SIZE)
        # print([self.view[i:i+2*self.vision+1] for i in range(0, (2*self.vision+1)**2, 2*self.vision+1)])
        self.view.append(pos[0]-pos_cat[0])
        self.view.append(pos[1]-pos_cat[1])
        return(self.view)
        
                     
    def pos_isValid(self, pos):
        if pos[0] >= self.plateau.n_cols or pos[0] < 0:
            return False
        if pos[1] >= self.plateau.n_rows or pos[1] < 0:
            return False
        return True
  
    # Check if the game is finished
    def is_done(self, cat):
        return cat.hasEaten(self)

    # Get the reward for the current state
    def get_reward(self, cat):
        pos = int(self.pos[1]/size.BLOCK_SIZE), int(self.pos[0]/size.BLOCK_SIZE)
        pos_cat = int(cat.pos[1]/size.BLOCK_SIZE), int(cat.pos[0]/size.BLOCK_SIZE)
        position = (pos[0]+pos_cat[0])**2+(pos[1]+pos_cat[1])**2
        return math.log(0.1+position)
    
    def get_actions(self):
            return [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]

    # Take an action and return the next state of the game
    def take_action(self , number_action, cat):
        # action = [0] * 5
        # action[random.randint(0, 4)] = 1
        action = self.get_actions()[number_action]
        self.move(action)
        return self.get_state(cat)