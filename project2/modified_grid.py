from grid import Grid
import random
class ModifiedGrid(Grid):

    def __init__(self, shape=(5,5)) -> None:
        super().__init__(shape)

        self.terminal_states = [(2,4),(4,0)]
        self.red_pos = (4,2)
        self.color_tiles = [self.red_pos,self.blue_pos,self.green_pos,self.green_pos,self.yellow_pos,self.terminal_states[0],self.terminal_states[1]]
    def move(self, action):
        self.current_state,reward =  super().move(action)
        termination = self.current_state in self.terminal_states 
        if reward==0:
            if self.current_state in self.color_tiles:
                return self.current_state,reward,termination
            else:
                return self.current_state,reward-0.2,termination
        else:
            return self.current_state,reward,termination
        



    def permute(self):
        if random.random() < 0.1 : 
            self.red_pos,self.blue_pos =self.blue_pos,self.red_pos
    
    