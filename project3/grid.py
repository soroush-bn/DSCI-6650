#todo 

import numpy as np

import random

class Grid:

    def __init__(self,shape=(5,5)) -> None:
        
        self.shape = shape
        self.action_set = ["left","right","up","down"]
        self.grid = np.zeros(self.shape)
        self.current_state = (4,0)
        self.blue_pos = (4,0)
        self.red_states = [(2,0),(2,1),(2,3),(2,4)]
        self.terminal_states= [(0,0),(0,4)]
        # self.color_map = {self.blue_pos: "blue", self.green_pos: "green" ,self.yellow_pos:"yellow", self.red_pos:"red"}
    
    def move(self,action):
        reward = self.check_special() #this step reward or total reward ???
        termination = self.current_state in self.terminal_states 

        if reward!=0 :
            return self.current_state, reward, termination

        if action== "right":
            
            if self.check_edge_right():
                return self.current_state, -1,termination
            else:
                self.current_state = (self.current_state[0],self.current_state[1]+1)
                # reward = self.check_special()
                return self.current_state,reward-1,termination
        elif action== "left":
            
            if self.check_edge_left():
                return self.current_state, -1,termination
            else:
                self.current_state = (self.current_state[0],self.current_state[1]-1)
                # reward = self.check_special()
                return self.current_state,reward-1,termination

        elif action== "up":
            
            if self.check_edge_top():
                return self.current_state, -1,termination
            else:
                self.current_state = (self.current_state[0]-1,self.current_state[1])
                # reward = self.check_special()
                return self.current_state,reward-1,termination
            
        elif action== "down":
            
            if self.check_edge_down():
                return self.current_state, -1,termination
            else:
                self.current_state = (self.current_state[0]+1,self.current_state[1])
                # reward = self.check_special()
                return self.current_state,reward-1,termination
        else:
            raise Exception("unknown action")
                


    

    def check_special(self):
        if self.current_state in self.red_states:
            # jump to red
            self.current_state= self.blue_pos
            return -20
        #green
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
        
