from plateau import Plateau, Case
from cat import Cat
from mouse import Mouse
import pygame
from utils import colors, size
from mouseState import MouseState
from catState import CatState

# Define the game state class
class Game:
    def __init__(self, n_cols, n_rows):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.plateau = Plateau(n_cols, n_rows)
        self.mouse = MouseState(0, self.plateau)
        self.cat = CatState( n_cols* n_rows -1, self.plateau)
        self.nb_step = 0
        SCREEN_SIZE = (n_cols*size.BLOCK_SIZE, n_rows*size.BLOCK_SIZE)
        self.screen =  pygame.display.set_mode(SCREEN_SIZE)
        #self.screen = pygame.display.set_mode((n_cols*size.BLOCK_SIZE, n_rows*size.BLOCK_SIZE))
        pygame.display.set_caption('Chase Tag')
        
    def step(self):
        self.nb_step +=1
        if self.nb_step % 2 == 0 :
            self.mouse.move_with_keyboard()
            self.mouse.get_state(self.cat)
        else : 
            self.cat.move_with_keyboard()
            self.cat.get_state(self.mouse)
        # print(self.cat.pos)
        
        self.plateau.draw_plateau(self.screen) 
        self.cat.draw_cat(self.screen)
        self.mouse.draw_mouse(self.screen)
        pygame.display.update()
        return self.cat.hasEaten(self.mouse)
    
if __name__ == '__main__':
    game = Game(15, 15)
    
    #game loop
    while True:
        game_over = game.step()
        game_over = game.step()
    #    sdddddqqq
        if game_over == True:
            break     
    pygame.quit()
        
        
    