from plateau import Plateau, Case
from cat import Cat
from mouse import Mouse
import pygame
from utils import colors, size
from mouseState import MouseState
from catState import CatState, Cats

# Define the environment class
class GameEnv:
    def __init__(self, n_cols, n_rows, number_of_cats = 3, vision = 0):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.vision = vision
        self.number_of_cats = number_of_cats
        self.plateau = Plateau(n_cols, n_rows)
        self.mouse = MouseState(0, self.plateau, vision = vision)
        self.cats = Cats( n_cols* n_rows -1, self.plateau, number_of_cats,vision = vision)
        self.cat_observation_space = self.cats.observation_space
        self.cat_action_space = self.cats.action_space
        self.mouse_observation_space = self.mouse.observation_space
        self.mouse_action_space = self.mouse.action_space
        self.nb_step = 0
        SCREEN_SIZE = (n_cols*size.BLOCK_SIZE, n_rows*size.BLOCK_SIZE)
        self.screen =  pygame.display.set_mode(SCREEN_SIZE)
        #self.screen = pygame.display.set_mode((n_cols*size.BLOCK_SIZE, n_rows*size.BLOCK_SIZE))
        pygame.display.set_caption('Chase Tag')
    
    # Reset the environment to its initial state
    def reset(self):
        self.mouse = MouseState(0, self.plateau, vision = self.vision)
        self.cats = Cats( self.n_cols* self.n_rows -1, self.plateau, self.number_of_cats,vision = self.vision)
        return self.cats.get_state(self.mouse), self.mouse.get_state(self.cat)
    
    def reset_cat(self):
        self.mouse = MouseState(0, self.plateau, vision = self.vision)
        self.cats = Cats( self.n_cols* self.n_rows -1, self.plateau, self.number_of_cats,vision = self.vision)
        return self.cats.get_state(self.mouse)
    
    def reset_mouse(self):
        self.mouse = MouseState(0, self.plateau, vision = self.vision)
        self.cats = Cats( self.n_cols* self.n_rows -1, self.plateau, self.number_of_cats,vision = self.vision)
        return self.mouse.get_state(self.cats.tabCats[0])
    
    def draw(self):
        self.plateau.draw_plateau(self.screen) 
        self.cats.draw_cats(self.screen)
        self.mouse.draw_mouse(self.screen)
        pygame.display.update()

    # Step the environment by taking an action
    def cat_step(self, action, i):
        self.nb_step += 1
        next_state_cat = self.cats.tabCats[i].take_action(action, self.mouse)
        done = self.cats.tabCats[i].is_done(self.mouse)
        reward = self.cats.tabCats[i].get_reward(self.mouse)
        self.cat_state = next_state_cat
        return next_state_cat, reward, done, {}
    
    def mouse_step(self, action):
        next_state_mouse = self.mouse.take_action(action, self.cats.tabCats[0])
        done = self.mouse.is_done(self.cats.tabCats[0])
        reward = self.mouse.get_reward(self.cats.tabCats[0])
        self.mouse_state = next_state_mouse
        return next_state_mouse, reward, done, {}
        
        
# if __name__ == '__main__':
#     game = GameEnv(6, 6)
    
#     #game loop
#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 break

#         next_state, reward, game_over, dic = game.step()
#         game.draw()
#     #   
#         if game_over == True:
#             break     
#     pygame.quit()