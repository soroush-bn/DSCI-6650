#todo 

import numpy as np

import random

class Grid:

    def __init__(self,shape=(7,7)) -> None:
        
        self.shape = shape
        self.action_set = ["left","right","up","down"]
        self.grid = np.zeros(self.shape)
        self.current_state = (3,3)
        # self.blue_pos = (4,0)
        # self.red_states = [(2,0),(2,1),(2,3),(2,4)]
        # self.terminal_states= [(0,0),(0,4)]
        self.upper_right = (0,6) # +1
        self.lower_left = (6,0) # -1

        # self.color_map = {self.blue_pos: "blue", self.green_pos: "green" ,self.yellow_pos:"yellow", self.red_pos:"red"}
    
    def move(self,action):
        # reward = self.check_special() #this step reward or total reward ???
        # termination = self.current_state == self.upper_right or  self.current_state == self.lower_left
        termination = False
        reward = 0 
        if  self.current_state == self.upper_right :
            termination= True
            reward =1
        elif self.current_state == self.lower_left :
            termination= True
            reward =-1

            
        # if reward!=0 :
        #     return self.current_state, reward, termination

        if action== "right":
            
            if self.check_edge_right():
                return self.current_state, 0,termination
            else:
                self.current_state = (self.current_state[0],self.current_state[1]+1)
                # reward = self.check_special()
                return self.current_state,reward,termination
        elif action== "left":
            
            if self.check_edge_left():
                return self.current_state, 0,termination
            else:
                self.current_state = (self.current_state[0],self.current_state[1]-1)
                # reward = self.check_special()
                return self.current_state,reward,termination

        elif action== "up":
            
            if self.check_edge_top():
                return self.current_state, 0,termination
            else:
                self.current_state = (self.current_state[0]-1,self.current_state[1])
                # reward = self.check_special()
                return self.current_state,reward,termination
            
        elif action== "down":
            
            if self.check_edge_down():
                return self.current_state, 0,termination
            else:
                self.current_state = (self.current_state[0]+1,self.current_state[1])
                # reward = self.check_special()
                return self.current_state,reward,termination
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
        
