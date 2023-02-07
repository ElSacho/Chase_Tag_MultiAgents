from plateau import Plateau
import pygame
from utils import colors, size
import numpy as np


class Mouse:
    def __init__(self, starting_position, plateau):
        self.case_number = starting_position
        self.plateau = plateau
        self.pos = [(self.case_number % plateau.n_cols )*size.BLOCK_SIZE + size.BLOCK_SIZE/2, (self.case_number // plateau.n_rows) *size.BLOCK_SIZE + size.BLOCK_SIZE/2]
        self.isDead = False
        self.step = 0
        
    def move_with_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.unicode == 'z':
                    self.move([0,1,0,0,0])
                elif event.unicode == 's':
                    self.move([0,0,0,1,0])
                elif event.unicode == 'q':
                    self.move([0,0,1,0,0])
                elif event.unicode == 'w':
                    self.move([1,0,0,0,0])
           
    def move(self, action):
        # ne rien faire
        if action == [0,0,0,0,1]:
            pass
        # aller à droite
        elif action == [0,0,0,1,0]:
            self.move_right()
        # aller à gauche
        elif action == [0,0,1,0,0]:
            self.move_left()
        # aller en haut
        elif action == [0,1,0,0,0]:
            self.move_up()
        # aller en bas
        elif action == [1,0,0,0,0]:
            self.move_down()
        self.step += 1
            
    def update_case_number(self):
        x, y = int(self.pos[1]/size.BLOCK_SIZE), int(self.pos[0]/size.BLOCK_SIZE)
        self.case_number = x*self.plateau.n_cols + y
        if self.plateau.cases[self.case_number].timeToSpend == 0:
            self.pos = [(self.case_number % self.plateau.n_cols )*size.BLOCK_SIZE + size.BLOCK_SIZE/2, (self.case_number // self.plateau.n_rows) *size.BLOCK_SIZE + size.BLOCK_SIZE/2]
        
    def move_down(self):
        if self.case_number + self.plateau.n_cols < self.plateau.n_cols * self.plateau.n_rows:
            lenght_mouvement = 1 / (self.plateau.cases[self.case_number].timeToSpend + 1)
            self.pos[1] += lenght_mouvement * size.BLOCK_SIZE
            self.update_case_number()
            
    def move_up(self):
        if self.case_number - self.plateau.n_cols >= 0:
            lenght_mouvement = 1 / (self.plateau.cases[self.case_number].timeToSpend + 1)
            self.pos[1] -= lenght_mouvement * size.BLOCK_SIZE
            self.update_case_number()
                  
    def move_right(self):
        if (self.case_number+1) % self.plateau.n_cols != 0:
            lenght_mouvement = 1 / (self.plateau.cases[self.case_number].timeToSpend + 1)
            self.pos[0] += lenght_mouvement * size.BLOCK_SIZE
            self.update_case_number()
                   
    def move_left(self):
        if (self.case_number) % self.plateau.n_cols != 0:
            lenght_mouvement = 1 / (self.plateau.cases[self.case_number].timeToSpend + 1)
            self.pos[0] -= lenght_mouvement * size.BLOCK_SIZE
            self.update_case_number()
              
    def analyse_ralentisseur(self):
        print(self.time_to_spend_on_ralentisseur)
        print(f'case : {self.plateau.cases[self.case_number].timeToSpend}')
        # Si on est sur une case classique, il ne se passe rien, on est pas sur un ralentisseur
        if self.plateau.cases[self.case_number].timeToSpend == 0:
            self.isOnRalentisseur = False
            return 0
        # Sinon si on est sur une case ralentisseur mais qu'on ne doit plus attendre, on avance 
        elif self.time_to_spend_on_ralentisseur == 0 and self.isOnRalentisseur:
            self.isOnRalentisseur = False
            return 0
        # Sinon si on vient d'arriver sur une case ralentisseur, on met a jour le temps d'attente et l etat du chat
        elif self.plateau.cases[self.case_number].timeToSpend !=0 and not self.isOnRalentisseur:
            self.isOnRalentisseur = True
            self.time_to_spend_on_ralentisseur = self.plateau.cases[self.case_number].timeToSpend
            print(f"time : {self.time_to_spend_on_ralentisseur}")
        # Sinon on est juste en attente et on reste la un tempo de plus
        else : 
            self.time_to_spend_on_ralentisseur -= 1
        print(self.time_to_spend_on_ralentisseur)
        # On renvoie le temps d'attente final
        return self.time_to_spend_on_ralentisseur
    
    def draw_mouse(self, screen):
        pygame.draw.circle (screen, colors.BLUE2, (self.pos[0], self.pos[1]), size.BLOCK_SIZE/3, 0)   
