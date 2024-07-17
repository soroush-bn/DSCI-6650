import numpy as np

import random

class Grid:

    def __init__(self,shape=(5,5)) -> None:
        
        self.shape = shape
        self.action_set = ["left","right","up","down"]
        self.grid = np.zeros(self.shape)
        self.current_state = (0,0)
        self.blue_pos = (0,1)
        self.green_pos = (0,4)
        self.red_pos = (3,2)
        self.yellow_pos = (4,4)
        self.color_map = {self.blue_pos: "blue", self.green_pos: "green" ,self.yellow_pos:"yellow", self.red_pos:"red"}
    
    def move(self,action):
        reward = self.check_special()
        if reward!=0 :
            return self.current_state, reward

        if action== "right":
            
            if self.check_edge_right():
                return self.current_state, -0.5
            else:
                self.current_state = (self.current_state[0],self.current_state[1]+1)
                # reward = self.check_special()
                return self.current_state,reward
        elif action== "left":
            
            if self.check_edge_left():
                return self.current_state, -0.5
            else:
                self.current_state = (self.current_state[0],self.current_state[1]-1)
                # reward = self.check_special()
                return self.current_state,reward

        elif action== "up":
            
            if self.check_edge_top():
                return self.current_state, -0.5
            else:
                self.current_state = (self.current_state[0]-1,self.current_state[1])
                # reward = self.check_special()
                return self.current_state,reward
            
        elif action== "down":
            
            if self.check_edge_down():
                return self.current_state, -0.5
            else:
                self.current_state = (self.current_state[0]+1,self.current_state[1])
                # reward = self.check_special()
                return self.current_state,reward
        else:
            raise Exception("unknown action")
                


    

    def check_special(self):
        #blue 
        if self.current_state[0] == self.blue_pos[0] and self.current_state[1]==self.blue_pos[1]:
            # jump to red
            self.current_state= self.red_pos
            return 5
        #green
        elif self.current_state[0] == self.green_pos[0] and self.current_state[1]==self.green_pos[1]:
            self.current_state = random.choice([self.red_pos,self.yellow_pos])
            return 2.5
        return 0
        
    def check_edge_right(self):
        if self.current_state[1]==self.shape[1]-1:
            return True
    
    def check_edge_left(self):
        if self.current_state[1]==0:
            return True
    
    def check_edge_top(self):
        if self.current_state[0]==0:
            return True
    
    def check_edge_down(self):
        if self.current_state[0]==self.shape[0]-1:
            return True
        
