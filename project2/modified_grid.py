from grid import Grid
import random
class ModifiedGrid(Grid):

    def __init__(self, shape=(5,5)) -> None:
        super().__init__(shape)
        self.permute_enable =False
        self.terminal_states = [(2,4),(4,0)]
        self.red_pos = (4,2)
        self.color_tiles = [self.red_pos,self.blue_pos,self.green_pos,self.green_pos,self.yellow_pos,self.terminal_states[0],self.terminal_states[1]]
        self.color_map = {self.blue_pos: "blue", self.green_pos: "green" ,self.yellow_pos:"yellow", self.red_pos:"red",(4,0):"black",(2,4):"black"}
    def move(self, action):
        
        self.current_state,reward =  super().move(action)
        termination = self.current_state in self.terminal_states 
        if self.permute_enable: self.permute()
        if reward==0:
            if self.current_state in self.color_tiles:
                return self.current_state,reward,termination
            else:
                return self.current_state,reward-0.2,termination
        else:
            return self.current_state,reward,termination
        



    def permute(self):
        if random.random() < 0.3 : 
            self.green_pos,self.blue_pos =self.blue_pos,self.green_pos
    
    