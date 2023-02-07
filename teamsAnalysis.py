'''
Jonathan Sosa 
Jan 2023
PEAK PERFORMANCE DATA ANALYSIS
'''

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#TeamAnalysis class
class TeamAnalysis():
    def __init__(self):
        '''TeamAnalysis constructor
        NOTE: the objects should be initialized already

        Parameters:
        -----------
        '''

        # fall: array of An objects
        #   This variable contains all the fall sports 
        self.fall= []

        # winter: array of An objects
        #   This variable contains all the winter sports 
        self.winter= []

        # spring: array of An objects
        #   This variable contains all the spring sports 
        self.spring= []
        
    #reset methods
    def fallReset(self):
        ''' This method is used to reset the array for the fall variable
        '''

        self.fall= []

    def winterReset(self):
        ''' This method is used to reset the array for the winter variable
        '''

        self.winter= []

    def springReset(self):
        ''' This method is used to reset the array for the spring variable
        '''

        self.spring= []

    #add methods
    def fallAdd(self,team):
        ''' This method is used to add another object to the array for the fall variable
        Parameters
        ------------------
        team: An object
        '''

        self.fall.append(team)

    def winterAdd(self,team):
        ''' This method is used to add another object to the array for the winter variable
        Parameters
        ------------------
        team: An object
        '''

        self.winter.append(team)

    def springAdd(self,team):
        ''' This method is used to add another object to the array for the spring variable
        Parameters
        ------------------
        team: An object
        '''

        self.spring.append(team)

    #initialize method









