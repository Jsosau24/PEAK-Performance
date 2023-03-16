'''
Jonathan Sosa 
Jan 2023
PEAK PERFORMANCE DATA ANALYSIS
HawkingDynamicsAnalysisTeams.py
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

        # fall_teams: array of An objects
        #   This variable contains all the fall sports analysis objects
        self.fall_teams= []

        # winter_teams: array of An objects
        #   This variable contains all the winter sports analysis objects
        self.winter_teams= []

        # spring_teams: array of An objects
        #   This variable contains all the spring sports analysis objects
        self.spring_teams= []

        # fall_data: array of An objects
        #   This variable contains all the fall sports data
        self.fall_data= None

        # winter_data: array of An objects
        #   This variable contains all the winter sports data
        self.winter_data= None

        # spring_data: array of An objects
        #   This variable contains all the spring sports data
        self.spring_data= None
        
    #reset teams methods
    def reset_fall_sports(self):
        ''' This method is used to reset the array for the fall variable
        '''

        self.fall_teams= []

    def reset_winter_sports(self):
        ''' This method is used to reset the array for the winter variable
        '''

        self.winter_teams= []

    def reset_spring_sports(self):
        ''' This method is used to reset the array for the spring variable
        '''

        self.spring_teams= []

    #add teams methods
    def add_fall_sports(self, team):
        ''' This method is used to add another object to the array for the fall variable
        Parameters
        ------------------
        team: An object
        '''

        self.fall_teams.append(team)

    def add_winter_sports(self, team):
        ''' This method is used to add another object to the array for the winter variable
        Parameters
        ------------------
        team: An object
        '''

        self.winter_teams.append(team)

    def add_spring_sports(self, team):
        ''' This method is used to add another object to the array for the spring variable
        Parameters
        ------------------
        team: An object
        '''

        self.spring_teams.append(team)

    #set teams methods
    def set_fall_sports(self, teams):
        ''' This method is used to set the fall_teams list
        Parameters
        ------------------
        team: list of analysis objects
        '''

        self.fall_teams = teams

    def set_winter_sports(self, teams):
        ''' This method is used to set the winter_teams list
        Parameters
        ------------------
        team: list of analysis objects
        '''

        self.winter_teams = teams
    
    def set_spring_sports(self, teams):
        ''' This method is used to set the spring_teams list
        Parameters
        ------------------
        team: list of analysis objects
        '''

        self.spring_teams = teams

    # helper methods
    def filter_df_by_date (self, beggining, end, data):
        '''Cleans the dataset so that we only have the data from the year we need as well as the
        Parameters:
        -----------
        beggining: str. format mm/dd/year
        end: str. format mm/dd/year
        data: pandas dataset
        
        Returns:
        -----------
        pandas datset
        '''

        return data.loc[(data['Date'] >= beggining) & (data['Date'] < end)]

    #initialize method
    def initialize(self):
        '''This method collets all the data form all the analyse objects and concatenates the data into 3 differnt pandas df depending on the teams season
        Parameters
        ------------------
        '''

        #creates an array with all the data for each sports season
        fall_datasets=[]
        winter_datasets=[]
        spring_datasets=[]

        for team in self.fall_teams:
            fall_datasets.append(team.filter_data)

        for team in self.winter_teams:
            winter_datasets.append(team.filter_data)

        for team in self.spring_teams:
            spring_datasets.append(team.filter_data)

        #concatenates the data for each sports season
        self.fall_data=pd.DataFrame(columns=fall_datasets[0].columns)
        for i in range(len(fall_datasets)):
            self.fall_data = self.fall_data.append(fall_datasets[i], ignore_index=True)

        self.winter_data=pd.DataFrame(columns=winter_datasets[0].columns)
        for i in range(len(winter_datasets)):
            self.winter_data = self.winter_data.append(winter_datasets[i], ignore_index=True)

        self.spring_data=pd.DataFrame(columns=spring_datasets[0].columns)
        for i in range(len(spring_datasets)):
            self.spring_data = self.spring_data.append(spring_datasets[i], ignore_index=True)

    #graphs
    def compare_teams_graphs(self):
        '''graphs that compares all e diferent sports trough the years

        Parameters
        ------------------
        '''
        #subplots
        fig, axs = plt.subplots(len(self.fall_teams[0].variables), 3, figsize=(20, 20))
        plt.subplots_adjust(hspace=0.8)

        cols=['Fall Sports','Winter Sports','Spring Sports']
        
        for ax, col in zip(axs[0], cols):
            ax.set_title(col)

        for ax, row in zip(axs[:,0], self.fall_teams[0].variables):
            ax.set_ylabel(row, size='large')

        for col in range(3): # 3 seasons

            if col == 0:
                ds = self.fall_data

            if col == 1:
                ds = self.winter_data

            if col == 2:
                ds = self.spring_data

            for row in range(len(self.fall_teams[0].variables)):

                #gets the dates for each data set (X variable)
                dates = ds.loc[:,'Date'] 

                #scatterplot
                axs[row,col].scatter(dates,ds.loc[:,self.fall_teams[0].variables[row]], label="Data Points", alpha=.3)

                #linear regression
                s = ds.set_index('Date')[self.fall_teams[0].variables[row]]

                y = s
                x = (s.index - pd.Timestamp(0)).days.values
                            
                #calculate equation for trendline
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)

                #add trendline to plot
                axs[row,col].plot(dates, p(x), color="purple", linewidth=1, linestyle="--", label="Trend")

                # plot the concatenated data and the trendline
                axs[row,col].scatter(x, y, label="Data Points", alpha=.3)
                

                #get the average of the variable
                avg = ds.loc[:,self.fall_teams[0].variables[row]].mean()
                mdc = self.fall_teams[0].get_minimal_detectable_change_of_a_dataframe(ds.loc[:,self.fall_teams[0].variables[row]])

                #define the top and bottom bounds
                topMDC = avg + mdc
                botMDC = avg - mdc

                #highlight area
                axs[row,col].fill_between(dates, topMDC, botMDC, color='C0', alpha=0.3, label='MDC')
                axs[row,col].fill_between(dates, p(x), botMDC, where=(p(x) < botMDC), color='C1', alpha=0.3, label='Under MDC')
                axs[row,col].fill_between(dates, topMDC, p(x), where=(p(x) > topMDC), color='C2', alpha=0.3, label='Over MDC')

                #intersection points
                trendY= p(x)

                #top MDC
                idx = np.argwhere(np.diff(np.sign(trendY - topMDC))).flatten()
                axs[row,col].plot(x[idx], trendY[idx], 'ro')

                #Bottom MDC
                idx = np.argwhere(np.diff(np.sign(trendY - botMDC))).flatten()
                axs[row,col].plot(x[idx], trendY[idx], 'ro')        
        
        #graph configuration
        plt.suptitle("test graph", fontsize=20, y=1)
        plt.gcf().autofmt_xdate()
        fig.tight_layout()
        plt.show()

    def fall_teams_graphs_per_season(self, title="Fall Teams by Season"):
        '''This method compares the preseason, post season, and season to each other
        Parameter
        ------------------
        '''

        #subplots
        fig, axs = plt.subplots(len(self.fall_teams[0].variables), 3, figsize=(20, 20))
        plt.subplots_adjust(hspace=0.8)

        cols=['Spring(Pre-Season)','Fall(Season)','Winter(Post-Season)']
        
        for ax, col in zip(axs[0], cols):
            ax.set_title(col)

        for ax, row in zip(axs[:,0], self.fall_teams[0].variables):
            ax.set_ylabel(row, size='large')

        for col in range(3): # 3 seasons

            if col == 0:
                #spring season
                ds = self.filter_df_by_date(self.fall_teams[0].spring_date,self.fall_teams[0].summer_date,self.fall_data)

            elif col == 1:
                #fall season
                ds = self.filter_df_by_date(self.fall_teams[0].fall_date,self.fall_teams[0].winter_date,self.fall_data)

            elif col == 2:
                #winter season
                ds = self.filter_df_by_date(self.fall_teams[0].winter_date,self.fall_teams[0].spring_date,self.fall_data)

            for row in range(len(self.fall_teams[0].variables)):

                #convert date column to datetime and subtract one week
                ds['Date-2week'] = ds['Date'] - pd.to_timedelta(14, unit='d')

                #calculate sum of values, grouped by week
                week2 = ds.groupby([pd.Grouper(key='Date-2week', freq='2W')])

                #convert date column to datetime and subtract one week
                ds['Date-week'] = ds['Date'] - pd.to_timedelta(7, unit='d')

                #calculate sum of values, grouped by week
                week = ds.groupby([pd.Grouper(key='Date-week', freq='W')])

                dates = ds.loc[:,'Date'] #gets the dates for each data set (X variable)

                #scatterplot
                axs[row,col].set_title(self.fall_teams[0].variables[row])
                axs[row,col].scatter(dates,ds.loc[:,self.fall_teams[0].variables[row]], label="Data Points", alpha=.3)

                #linear regression
                s = ds.set_index('Date')[self.fall_teams[0].variables[row]]

                y = s
                x = (s.index - pd.Timestamp(0)).days.values
                            
                #calculate equation for trendline
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)

                #add trendline to plot
                axs[row,col].plot(dates, p(x), color="purple", linewidth=1, linestyle="--", label="Trend")

                #gets the MDC
                count=0

                for wk, group in week2:

                    if count == 0:
                        count = count + 1

                    if count == 1:

                        count = count + 1
                    
                        #get the average of the variable
                        avgWK = group.loc[:,self.fall_teams[0].variables[row]].mean()
                        mdcWK = self.fall_teams[0].get_minimal_detectable_change_of_a_dataframe(group.loc[:,self.fall_teams[0].variables[row]])

                        #define the bottom bounds
                        botMDC = avgWK - mdcWK
                        topMDC = avgWK + mdcWK

                        #highlight area
                        axs[row,col].fill_between(dates, topMDC, botMDC, color='C0', alpha=0.3, label='MDC')
                        axs[row,col].fill_between(dates, p(x), botMDC, where=(p(x) < botMDC), color='C1', alpha=0.3, label='Under MDC')
                        axs[row,col].fill_between(dates, topMDC, p(x), where=(p(x) > topMDC), color='C2', alpha=0.3, label='Over MDC')

                        #intersection points
                        trendY= p(x)
                        #top MDC
                        idx = np.argwhere(np.diff(np.sign(trendY - topMDC))).flatten()
                        axs[row,col].plot(x[idx], trendY[idx], 'ro')
                        #Bottom MDC
                        idx = np.argwhere(np.diff(np.sign(trendY - botMDC))).flatten()
                        axs[row,col].plot(x[idx], trendY[idx], 'ro')

                        break
                    
                for wk, group in week:
                    axs[row,col].scatter(wk,group.loc[:,self.fall_teams[0].variables[row]].mean(),color = 'green')
        
        #graph configuration
        plt.suptitle(title, fontsize=20, y=1)
        plt.legend(["Data", "Trend"], loc ="lower right")
        plt.gcf().autofmt_xdate()
        fig.tight_layout()
        plt.show()


        









