"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 2  # grid height
MAZE_W = 5  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['cross_up', 'cross_down', 'straight']
        self.n_actions = len(self.action_space)
        self.title('maze')
        #self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.change = 0

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        '''for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)'''
            
        # create origin
        origin = np.array([20, 20])
        self.all_nodes = []
        
        for i in range(MAZE_W):
            
            if i == 0:
                node_center = origin + np.array([i, i])
                self.node = self.canvas.create_oval(
                    node_center[0] - 15, node_center[1] - 15,
                    node_center[0] + 15, node_center[1] + 15,
                    fill='green')
                
                node_down_center = origin + np.array([i, UNIT])
                self.node_down = self.canvas.create_oval(
                    node_down_center[0] - 15, node_down_center[1] - 15,
                    node_down_center[0] + 15, node_down_center[1] + 15,
                    fill='green')
                
                self.all_nodes.append(self.node)
                self.all_nodes.append(self.node_down)
            elif i == (MAZE_W-1):
                oval_center = origin + np.array([UNIT*i, 0])
                self.oval = self.canvas.create_oval(
                    oval_center[0] - 15, oval_center[1] - 15,
                    oval_center[0] + 15, oval_center[1] + 15,
                    fill='yellow')
                self.all_nodes.append(self.oval)
            else:
                node_center = origin + np.array([UNIT*i, 0])
                self.node = self.canvas.create_rectangle(
                    node_center[0] - 15, node_center[1] - 15,
                    node_center[0] + 15, node_center[1] + 15,
                    fill='black')
                
                node_down_center = origin + np.array([UNIT*i, UNIT])
                self.node_down = self.canvas.create_rectangle(
                    node_down_center[0] - 15, node_down_center[1] - 15,
                    node_down_center[0] + 15, node_down_center[1] + 15,
                    fill='black')
                
                self.all_nodes.append(self.node)
                self.all_nodes.append(self.node_down)
            

        '''# top left[5, 5, 35, 35]
        node1_center = origin + np.array([0, 0])
        self.node1 = self.canvas.create_oval(
            node1_center[0] - 15, node1_center[1] - 15,
            node1_center[0] + 15, node1_center[1] + 15,
            fill='black')
        
        # bottom left[5, 45, 35, 75]
        node2_center = origin + np.array([0, UNIT])
        self.node2 = self.canvas.create_oval(
            node2_center[0] - 15, node2_center[1] - 15,
            node2_center[0] + 15, node2_center[1] + 15,
            fill='black')
        
        # top 2[45, 5, 75, 35]
        node3_center = origin + np.array([UNIT, 0])
        self.node3 = self.canvas.create_rectangle(
            node3_center[0] - 15, node3_center[1] - 15,
            node3_center[0] + 15, node3_center[1] + 15,
            fill='black')
        
        # bottom 2[45, 45, 75, 75]
        node4_center = origin + np.array([UNIT, UNIT])
        self.node4 = self.canvas.create_rectangle(
            node4_center[0] - 15, node4_center[1] - 15,
            node4_center[0] + 15, node4_center[1] + 15,
            fill='black')
        
        # top 3[85, 5, 115, 35]
        node5_center = origin + np.array([UNIT*2, 0])
        self.node5 = self.canvas.create_rectangle(
            node5_center[0] - 15, node5_center[1] - 15,
            node5_center[0] + 15, node5_center[1] + 15,
            fill='black')
        
        # bottom 3[85, 45, 115, 35]
        node6_center = origin + np.array([UNIT*2, UNIT])
        self.node6 = self.canvas.create_rectangle(
            node6_center[0] - 15, node6_center[1] - 15,
            node6_center[0] + 15, node6_center[1] + 15,
            fill='black')
        
        # top 4[125, 5, 155, 35]
        node7_center = origin + np.array([UNIT*3, 0])
        self.node7 = self.canvas.create_rectangle(
            node7_center[0] - 15, node7_center[1] - 15,
            node7_center[0] + 15, node7_center[1] + 15,
            fill='black')
        
        # bottom 4[125, 45, 155, 75]
        node8_center = origin + np.array([UNIT*3, UNIT])
        self.node8 = self.canvas.create_rectangle(
            node8_center[0] - 15, node8_center[1] - 15,
            node8_center[0] + 15, node8_center[1] + 15,
            fill='black')
        
        # Goal
        oval_center = origin + np.array([UNIT*4, 0])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')'''
        
        # create red oval
        self.rect = self.canvas.create_oval(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        
        origin = np.array([20, 20])
        '''node1_center = origin + np.array([0, UNIT])
        self.rect = self.canvas.create_oval(
            node1_center[0] - 15, node1_center[1] - 15,
            node1_center[0] + 15, node1_center[1] + 15,
            fill='red')'''
        if self.change == 0:
            self.change = 1
            node1_center = origin + np.array([0, 0])
            self.rect = self.canvas.create_oval(
                node1_center[0] - 15, node1_center[1] - 15,
                node1_center[0] + 15, node1_center[1] + 15,
                fill='red')
        else:
            self.change = 0
            node2_center = origin + np.array([0, UNIT])
            self.rect = self.canvas.create_oval(
                node2_center[0] - 15, node2_center[1] - 15,
                node2_center[0] + 15, node2_center[1] + 15,
                fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        #print("ddd {}".format(s))
        base_action = np.array([0, 0])
        if action == 0:   # cross up
            if s[1] > UNIT and s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
                base_action[1] -= UNIT
                print("cross up")
        elif action == 1:   # cross down
            if s[1] < (MAZE_H - 1) * UNIT and s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
                base_action[1] += UNIT
                print("cross down")
        elif action == 2:   # straight
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
                print("straight")

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state
        print("Next state {}".format(s_))
        print(self.all_nodes[4])
        # reward function
        if s_ == self.canvas.coords(self.all_nodes[-1]):
            reward = 0
            done = True
            s_ = 'terminal'
        #elif s_ in [self.canvas.coords(self.node3), self.canvas.coords(self.node5), self.canvas.coords(self.node7)]:
        elif s_ in [self.canvas.coords(self.all_nodes[2]), self.canvas.coords(self.all_nodes[4]), self.canvas.coords(self.all_nodes[6])]:
            reward = 1
            done = False
        #elif s_ in [self.canvas.coords(self.node4), self.canvas.coords(self.node6), self.canvas.coords(self.node8)]:
        elif s_ in [self.canvas.coords(self.all_nodes[3]), self.canvas.coords(self.all_nodes[5]), self.canvas.coords(self.all_nodes[7])]:
            reward = -1
            done = False
        
        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


