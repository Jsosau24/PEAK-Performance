'''
Jonathan Sosa 
Jan 2023
PEAK PERFORMANCE DATA ANALYSIS
'''

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import dates as mpl_dates
from matplotlib.patches import Rectangle

#Analysis class
class Analysis():

    #init method
    def __init__(self, path=None, year=2020, fall=None, winter=None, spring=None,summer=None,variables = ['System Weight', 'Jump Height', 'Braking RFD','Time To Takeoff','mRSI', 'Peak Relative Propulsive Power']):
        '''An constructor

        Parameters:
        -----------
        path: string
        year: int
        fall: string
        winter: stirng
        spring: string
        summer: string
        variables: array[string]
        '''

        # year: int
        #   This int contains the inital year of the data we want to obtain. EX for season 2020 - 2021 we write 2020
        self.year = year

        # path: string
        #   This string contains the path to the dataset
        self.path = path

        # data: pandas dataset
        #   This variable contains the raw data without any clean up
        self.data = None

        #variables: array of strings
        #   This variable contains the names of the variables we are going to use
        self.variables= variables
        self.all_variables = ['Date', 'Name']
        self.all_variables.extend(self.variables)

        # filter_data: pandas dataset
        #   This variable contains the clean version of the data without any extra vsriables or observations
        self.filter_data = None 

        # filter_data_athletes: pands group object
        #   datasets grouped by athletes
        self.filter_data_athletes = None

        # fall_group: pands group object
        #   datasets grouped by athletes in the fall
        self.fall_group = None

        # winter_group: pands group object
        #   datasets grouped by athletes in the Winter
        self.winter_group = None

        # spring_group: pands group object
        #   datasets grouped by athletes in the Spring
        self.spring_group = None       

        # fall_date: string
        #   contains day and month for the beggining fall season month/day format
        self.fall_date = fall

        # winter_date: string
        #   contains day and month for the beggining winter season month/day format
        self.winter_date = winter

        # spring_date: string
        #   contains day and month for the beggining spring season month/day format
        self.spring_date = spring

        # summer: string
        #   contains day and month for the beggining summer season month/day format
        self.summer_date = summer

        # fall_season_data
        #   This variable contains the observations of the filreted dataset for the Fall season
        self.fall_season_data = None

        # winter_season_data
        #   This variable contains the observations of the filreted dataset for the Winter season
        self.winter_season_data = None

        # spring_season_data
        #   This variable contains the observations of the filreted dataset for the Spring season
        self.spring_season_data = None

        # number_of_observations: int. 
        #   This variable contains the number of observaions
        self.number_of_observations = None

        # number_of_variables: int. 
        #   This variable contains the number of variables
        self.number_of_variables = None

        # decriptive_statistics: arr
        #   This variable conatians an array with descriptive statistics for quantitative variables
        self.decriptive_statistics = None

        # mean: arr
        #   This variable contains an array with the mean for the number of quantitative variables
        self.mean = None

        # standard_deviation: arr
        #   This variable contains an array with the Standard Deviation for the number of quantitative variables
        self.standard_deviation = None

    #set methods
    def set_path(self, path):
        """Replaces path instance variable with `path`.
        NOTE: you have to initialize the class to update

        Parameters:
        -----------
        path: string
        """

        self.path = path

    def set_fall_date(self, fall):
        """Replaces fall instance variable with `fall`.
        NOTE: you have to initialize the class to update

        Parameters:
        -----------
        fall: string
        """

        self.fall_date = fall

    def set_winter_date(self, winter):
        """Replaces winter instance variable with `winter`.
        NOTE: you have to initialize the class to update

        Parameters:
        -----------
        winter: string
        """

        self.winter_date = winter

    def set_spring_date(self, spring):
        """Replaces spring instance variable with `spring`.
        NOTE: you have to initialize the class to update

        Parameters:
        -----------
        spring: string
        """

        self.spring_date = spring

    def set_summer_date(self, summer):
        """Replaces summer instance variable with `summer`.
        NOTE: you have to initialize the class to update

        Parameters:
        -----------
        summer: string
        """

        self.summer_date = summer

    #initialize method
    def initialize(self):
        """Initialize the class by cleaning the data and start updating the variables for the class
        """

        #loads the data set into a pandas dataframe
        self.data = pd.read_csv(self.path)

        # Convert the Date to datetime64
        self.data['Date'] = pd.to_datetime(self.data['Date']) #format='%m/%d/%Y'

        #divides the datasets into the different seasons during the school year
        self.fall_date= pd.to_datetime(self.fall_date+'/'+str(self.year))
        self.winter_date= pd.to_datetime(self.winter_date+'/'+str(self.year))
        self.spring_date= pd.to_datetime(self.spring_date+'/'+str(self.year+1))
        self.summer_date= pd.to_datetime(self.summer_date+'/'+str(self.year+1))

        #calls the method to clean the data set
        self.clean_data()

        #sorts dataset by date
        self.filter_data=self.filter_data.sort_values(by='Date',ascending=True)

        #divides the dataset into seasons
        self.fall_season_data = self.filter_df_by_date(self.fall_date, self.winter_date, self.filter_data)
        self.winter_season_data = self.filter_df_by_date(self.winter_date, self.spring_date, self.filter_data)
        self.spring_season_data = self.filter_df_by_date(self.spring_date, self.summer_date, self.filter_data)

        #groups the data for individual atheltes trough the acedemic year
        self.filter_data_athletes = self.group_df_by_athlete(self.filter_data)

        #groups the data for individual athletes in each season
        self.fall_group= self.group_df_by_athlete(self.fall_season_data)
        self.winter_group= self.group_df_by_athlete(self.winter_season_data)
        self.spring_group= self.group_df_by_athlete(self.spring_season_data)

        #sets the reminder of the variables
        self.number_of_observations,self.number_of_variables = self.filter_data.shape
        self.decriptive_statistics = self.filter_data.describe()
        self.mean = self.filter_data.mean()
        self.standard_deviation = self.filter_data.std()

        #sets a column for weakly and bi-weekly averages
        self.filter_data['Date-2week'] = self.filter_data['Date'] - pd.to_timedelta(14, unit='d')
        self.filter_data['Date-week'] = self.filter_data['Date'] - pd.to_timedelta(7, unit='d')

        return

    #helper methods for initiallize methods
    def group_df_by_athlete (self, ds):
        '''Creates an array with individual datasets grouped by athlete
        Parameters:
        -----------
        ds: pandas dataset

        Returns:
        -----------
        pandas group obj
        '''
        
        return ds.groupby('Name')
        
    def clean_data (self):
        '''Cleans the dataset so that we only have the data from the year we need as well as the observations
            we need. This is saved to self.filter_data
        Parameters:
        -----------
        
        Returns:
        -----------
        array
        '''

        #filters the columns to the ones we are going to use
        self.filter_data = self.data[self.all_variables]

        #keeps only the data from that academic year
        self.filter_data = self.filter_df_by_date(self.fall_date, self.summer_date, self.filter_data)

        #drops all nan values
        self.filter_data.dropna()

        #removes noice from the data
        for i in range(len(self.variables)):

            #get the .5 and 95 quartile
            min = self.filter_data.loc[:,self.variables[i]].quantile(0.01)
            max = self.filter_data.loc[:,self.variables[i]].quantile(0.99)

            for x in self.filter_data.index:
                
                #sets the value to the defined quartile if the vallue exceeds that quartile
                if self.filter_data.loc[x, self.variables[i]] > max:
                    self.filter_data.loc[x, self.variables[i]] = max
                
                elif self.filter_data.loc[x, self.variables[i]] < min:
                    self.filter_data.loc[x, self.variables[i]] = min

        return self.filter_data

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

    #graphs methods
    def mdc_graphs_for_season_data(self, ds, title):
        '''Creates the graphs of every variable using the MDC of the second week as a baseline
        Parameters:
        -----------
        ds: pandas dataset
        title: string
        '''

        #convert date column to datetime and subtract one week
        ds['Date-2week'] = ds['Date'] - pd.to_timedelta(14, unit='d')

        #calculate sum of values, grouped by week
        week2 = ds.groupby([pd.Grouper(key='Date-2week', freq='2W')])

        #convert date column to datetime and subtract one week
        ds['Date-week'] = ds['Date'] - pd.to_timedelta(7, unit='d')

        #calculate sum of values, grouped by week
        week = ds.groupby([pd.Grouper(key='Date-week', freq='W')])

        dates = ds.loc[:,'Date']
        #subplots
        fig, axs = plt.subplots(len(self.variables), 1, figsize=(20, 40))
        plt.subplots_adjust(hspace=0.8)

        for i in range(len(self.variables)):

            #scatterplot
            axs[i].set_title(self.variables[i])
            axs[i].scatter(dates,ds.loc[:,self.variables[i]], label="Data Points", alpha=.3)

            #linear regression
            s = ds.set_index('Date')[self.variables[i]]

            y = s
            x = (s.index - pd.Timestamp(0)).days.values
                    
            #calculate equation for trendline
            z = np.polyfit(x, y, 10)
            p = np.poly1d(z)

            #add trendline to plot
            axs[i].plot(dates, p(x), color="purple", linewidth=1, linestyle="--", label="Trend")

            #gets the MDC
            count=0

            for wk, group in week2:

                if count == 0:
                    count = count + 1

                if count == 1:

                    count = count + 1
            
                    #get the average of the variable
                    avgWK = group.loc[:,self.variables[i]].mean()
                    mdcWK = self.get_minimal_detectable_change_of_a_dataframe(group.loc[:,self.variables[i]])

                    #define the bottom bounds
                    botMDC = avgWK - mdcWK
                    topMDC = avgWK + mdcWK

                    #highlight area
                    axs[i].fill_between(dates, topMDC, botMDC, color='C0', alpha=0.3, label='MDC')
                    axs[i].fill_between(dates, p(x), botMDC, where=(p(x) < botMDC), color='C1', alpha=0.3, label='Under MDC')
                    axs[i].fill_between(dates, topMDC, p(x), where=(p(x) > topMDC), color='C2', alpha=0.3, label='Over MDC')

                    #intersection points
                    trendY= p(x)
                    #top MDC
                    idx = np.argwhere(np.diff(np.sign(trendY - topMDC))).flatten()
                    axs[i].plot(x[idx], trendY[idx], 'ro')
                    #Bottom MDC
                    idx = np.argwhere(np.diff(np.sign(trendY - botMDC))).flatten()
                    axs[i].plot(x[idx], trendY[idx], 'ro')

                elif count > 0:
                    break
            
            for wk, group in week:
                axs[i].scatter(wk,group.loc[:,self.variables[i]].mean(),color = 'green')

        #graph configuration
        plt.suptitle(title, fontsize=20, y=1)
        plt.legend(loc ="lower right")
        plt.gcf().autofmt_xdate()
        fig.tight_layout()
        plt.show()

    #year round graphs
    def mdc_graphs_for_year_data (self, ds, title):
        '''Creates the graphs of every variable using the MDC of the whole year
        Parameters:
        -----------
        ds: pandas dataset
        title: string
        '''

        week2 = ds.groupby([pd.Grouper(key='Date-2week', freq='2W')])
        week = ds.groupby([pd.Grouper(key='Date-week', freq='W')])

        dates = ds.loc[:,'Date']

        #subplots
        fig, axs = plt.subplots(len(self.variables), 1, figsize=(20, 40))
        plt.subplots_adjust(hspace=0.8)

        for i in range(len(self.variables)):

            #get the average of the variable
            avg = ds.loc[:,self.variables[i]].mean()
            mdc = self.get_minimal_detectable_change_of_a_dataframe(ds.loc[:,self.variables[i]])

            #define the top and bottom bounds
            topMDC = avg + mdc
            botMDC = avg - mdc

            #scatterplot
            axs[i].set_title(self.variables[i])
            axs[i].scatter(dates,ds.loc[:,self.variables[i]], label="Data Points")

            #linear regression
            s = ds.set_index('Date')[self.variables[i]]

            y = s
            x = (s.index - pd.Timestamp(0)).days.values
                    
            #calculate equation for trendline
            z = np.polyfit(x, y, 10)
            p = np.poly1d(z)

            #add trendline to plot
            axs[i].plot(dates, p(x), color="purple", linewidth=1, linestyle="--", label="Trend")

            #highlight area
            axs[i].fill_between(dates, topMDC, botMDC, color='C0', alpha=0.3, label='MDC')
            axs[i].fill_between(dates, p(x), botMDC, where=(p(x) < botMDC), color='C1', alpha=0.3, label='Under MDC')
            axs[i].fill_between(dates, topMDC, p(x), where=(p(x) > topMDC), color='C2', alpha=0.3, label='Over MDC')

            #intersection points
            trendY= p(x)

            #top MDC
            idx = np.argwhere(np.diff(np.sign(trendY - topMDC))).flatten()
            axs[i].plot(x[idx], trendY[idx], 'ro')

            #Bottom MDC
            idx = np.argwhere(np.diff(np.sign(trendY - botMDC))).flatten()
            axs[i].plot(x[idx], trendY[idx], 'ro')

            
            #MDC 2 week
            for wk, group in week2:
            
                #get the average of the variable
                avgWK = group.loc[:,self.variables[i]].mean()
                mdcWK = self.get_minimal_detectable_change_of_a_dataframe(group.loc[:,self.variables[i]])

                #define the bottom bounds
                botMDCwk = avgWK - mdcWK

                axs[i].add_patch(Rectangle((wk, botMDCwk), pd.to_timedelta(14, unit='d'), (2*mdcWK),edgecolor='red',facecolor='none'))
            
            for wk, group in week:
                axs[i].scatter(wk,group.loc[:,self.variables[i]].mean(),color = 'hotpink')

        #graph configuration
        plt.suptitle(title, fontsize=20, y=1)
        plt.legend(loc ="lower right")
        plt.gcf().autofmt_xdate()
        fig.tight_layout()
        plt.show()
            
    def year_graphs_for_one_athlete (self, name):
        '''Creates the graphs of every variable for that person
        Parameters:
        -----------
        ds: pandas dataset
        '''
        
        ds = self.filter_data_athletes.get_group(name)
        self.mdc_graphs_for_year_data(ds, name)

    def year_graphs_for_all_athletes (self):
        '''Creates the graphs of every variable for that all individuals
        Parameters:
        -----------
        '''
        for name,group in self.filter_data_athletes:
            self.year_graphs_for_one_athlete(name)

    #seasonal graphs
    #fall
    def fall_season_graphs (self):
        '''Creates the graphs of every variables
        Parameters:
        -----------
        '''

        self.mdc_graphs_for_season_data(self.fall_season_data, 'Fall Season Graphs')

    def fall_graphs_for_one_athlete (self, name):
        '''Creates the graphs of every variables for an athlete
        Parameters:
        -----------
        name: string
        '''
        ds = self.fall_group.get_group(name)
        self.mdc_graphs_for_season_data(ds,name)

    def fall_graphs_for_all_athletes (self):
        '''Creates the graphs of every variable for that all individuals in the fall season
        Parameters:
        -----------
        '''

        for name,group in self.fall_group:
            self.fall_graphs_for_one_athlete(name)

    #winter
    def winter_season_graphs (self):
        '''Creates the graphs of every variables
        Parameters:
        -----------
        '''

        self.mdc_graphs_for_season_data(self.winter_season_data, 'Winter Team Data')

    def winter_graphs_for_one_athlete (self, name):
        '''Creates the graphs of every variables for an athlete
        Parameters:
        -----------
        name: string
        '''
        ds = self.winter_group.get_group(name)
        self.mdc_graphs_for_season_data(ds,name)

    def winter_graphs_for_all_athletes (self):
        '''Creates the graphs of every variable for that all individuals in the fall season
        Parameters:
        -----------
        '''

        for name,group in self.winter_group:
            self.winter_graphs_for_one_athlete(name)

    #spring
    def spring_season_graphs (self):
        '''Creates the graphs of every variables
        Parameters:
        -----------
        '''

        self.mdc_graphs_for_season_data(self.spring_season_data, 'Spring Team Data')

    def spring_graphs_for_one_athlete (self, name):
        '''Creates the graphs of every variables for an athlete
        Parameters:
        -----------
        name: string
        '''
        ds = self.spring_group.get_group(name)
        self.mdc_graphs_for_season_data(ds,name)

    def spring_graphs_for_all_athletes (self):
        '''Creates the graphs of every variable for that all individuals in the spring season
        Parameters:
        -----------
        '''

        for name,group in self.spring_group:
            self.spring_graphs_for_one_athlete(name)

    #team divided by season
    def team_graphs_by_season(self, title='Data in Different Seasons'):
        '''Creates the graphs of every variable divide by every season
        Parameters:
        -----------
        '''

        #subplots
        fig, axs = plt.subplots(len(self.variables), 3, figsize=(20, 20))
        plt.subplots_adjust(hspace=0.8)

        cols=['Fall','Winter','Spring']
        
        for ax, col in zip(axs[0], cols):
            ax.set_title(col)

        for ax, row in zip(axs[:,0], self.variables):
            ax.set_ylabel(row, size='large')

        for col in range(3): # 3 seasons

            if col == 0:
                ds = self.fall_season_data

            if col == 1:
                ds = self.winter_season_data

            if col == 2:
                ds = self.spring_season_data

            for row in range(len(self.variables)):

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
                axs[row,col].set_title(self.variables[row])
                axs[row,col].scatter(dates,ds.loc[:,self.variables[row]], label="Data Points", alpha=.3)

                #linear regression
                s = ds.set_index('Date')[self.variables[row]]

                y = s
                x = (s.index - pd.Timestamp(0)).days.values
                            
                #calculate equation for trendline
                z = np.polyfit(x, y, 10)
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
                        avgWK = group.loc[:,self.variables[row]].mean()
                        mdcWK = self.getMdcInd(group.loc[:,self.variables[row]])

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

                    elif count > 0:
                        break
                    
                for wk, group in week:
                    axs[row,col].scatter(wk,group.loc[:,self.variables[row]].mean(),color = 'green')
        
        #graph configuration
        plt.suptitle(title, fontsize=20, y=1)
        plt.legend(["Data", "Trend"], loc ="lower right")
        plt.gcf().autofmt_xdate()
        fig.tight_layout()
        plt.show()

    #get Statistics
    #mean
    def get_mean(self, variable):
        '''Returns the average for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.filter_data.loc[:,variable].mean()

    def get_mean_fall(self, variable):
        '''Returns the average for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fall_season_data.loc[:,variable].mean()

    def get_mean_winter(self, variable):
        '''Returns the average for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.winter_season_data.loc[:,variable].mean()

    def get_mean_spring(self, variable):
        '''Returns the average for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spring_season_data.loc[:,variable].mean()

    #count
    def get_count(self, variable):
        '''Returns the count for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.filter_data.loc[:,variable].count()

    def get_count_fall(self, variable):
        '''Returns the count for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fall_season_data.loc[:,variable].count()

    def get_count_winter(self, variable):
        '''Returns the count for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.winter_season_data.loc[:,variable].count()

    def get_count_spring(self, variable):
        '''Returns the count for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spring_season_data.loc[:,variable].count()

    #standard deviation
    def get_std(self, variable):
        '''Returns the standard deviation for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.filter_data.loc[:,variable].std()

    def get_std_fall(self, variable):
        '''Returns the standard deviation for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fall_season_data.loc[:,variable].std()
    
    def get_std_winter(self, variable):
        '''Returns the standard deviation for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.winter_season_data.loc[:,variable].std()

    def get_std_spring(self, variable):
        '''Returns the standard deviation for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spring_season_data.loc[:,variable].std()

    #the minimum value
    def get_minimun_value(self, variable):
        '''Returns the min for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.filter_data.loc[:,variable].min()

    def get_minimun_value_fall(self, variable):
        '''Returns the min for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fall_season_data.loc[:,variable].min()

    def get_minimun_value_winter(self, variable):
        '''Returns the min for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.winter_season_data.loc[:,variable].min()

    def get_minimun_value_spring(self, variable):
        '''Returns the min for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spring_season_data.loc[:,variable].min()

    #the maximum value
    def get_maximum_value(self, variable):
        '''Returns the max for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.filter_data.loc[:,variable].max()

    def get_maximum_value_fall(self, variable):
        '''Returns the max for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fall_season_data.loc[:,variable].max()

    def get_maximum_value_winter(self, variable):
        '''Returns the max for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.winter_season_data.loc[:,variable].max()

    def get_maximum_value_spring(self, variable):
        '''Returns the max for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spring_season_data.loc[:,variable].max()

    #the 25th quartile value
    def get_25_quartile(self, variable):
        '''Returns the 25th quartile value for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.filter_data.loc[:,variable].quantile(0.25)

    def get_25_quartile_fall(self, variable):
        '''Returns the 25th quartile value for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fall_season_data.loc[:,variable].quantile(0.25)

    def get_25_quartile_winter(self, variable):
        '''Returns the 25th quartile value for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.winter_season_data.loc[:,variable].quantile(0.25)

    def get_25_quartile_spring(self, variable):
        '''Returns the 25th quartile value for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spring_season_data.loc[:,variable].quantile(0.25)

    #the 50th quartile value
    def get_50_quartile(self, variable):
        '''Returns the 50th quartile value for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.filter_data.loc[:,variable].quantile(0.50)

    def get_50_quartile_fall(self, variable):
        '''Returns the 50th quartile value for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fall_season_data.loc[:,variable].quantile(0.50)

    def get_50_quartile_winter(self, variable):
        '''Returns the 50th quartile value for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.winter_season_data.loc[:,variable].quantile(0.50)

    def get_50_quartile_spring(self, variable):
        '''Returns the 50th quartile value for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spring_season_data.loc[:,variable].quantile(0.50) 

    #the 75th quartile value
    def get_70_quartile(self, variable):
        '''Returns the 75th quartile value for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.filter_data.loc[:,variable].quantile(0.75)

    def get_70_quartile_fall(self, variable):
        '''Returns the 75th quartile value for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fall_season_data.loc[:,variable].quantile(0.75)

    def get_70_quartile_winter(self, variable):
        '''Returns the 75th quartile value for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.winter_season_data.loc[:,variable].quantile(0.75)

    def get_70_quartile_spring(self, variable):
        '''Returns the 75th quartile value for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spring_season_data.loc[:,variable].quantile(0.75)

    #standard Error of measurement
    def get_standard_error_measurment(self, variable):
        '''Returns the SEM value for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.get_std(variable)/np.sqrt(self.get_count(variable)))

    def get_standard_error_measurment_fall(self, variable):
        '''Returns the SEM value for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.get_std_fall(variable)/np.sqrt(self.get_count_fall(variable)))

    def get_standard_error_measurment_winter(self, variable):
        '''Returns the SEM value for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.get_std_winter(variable)/np.sqrt(self.get_count_winter(variable)))

    def get_standard_error_measurment_spring(self, variable):
        '''Returns the SEM value for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.get_std_spring(variable)/np.sqrt(self.get_count_spring(variable)))

    def get_standard_error_measurment_of_a_dataframe(self, ds):
        '''Returns the SEM value for the stated variable for the dataset
        Parameters
        --------------
        ds: pandas DF
            column with with the desired data
        '''

        return (ds.std()/np.sqrt(ds.count()))

    #minimal detectable change
    def get_minimal_detectable_change(self, variable):
        '''Returns the MDC value for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        
        return (self.get_standard_error_measurment(variable)*1.96*np.sqrt(2))

    def get_minimal_detectable_change_fall(self, variable):
        '''Returns the MDC value for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.get_standard_error_measurment_fall(variable)*1.96*np.sqrt(2))

    def get_minimal_detectable_change_winter(self, variable):
        '''Returns the MDC value for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.get_standard_error_measurment_winter(variable)*1.96*np.sqrt(2))

    def get_minimal_detectable_change_sring(self, variable):
        '''Returns the MDC value for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.get_standard_error_measurment_spring(variable)*1.96*np.sqrt(2))

    def get_minimal_detectable_change_of_a_dataframe(self, col):
        '''Returns the SEM value for the stated variable for the dataset
        Parameters
        --------------
        col: pandas DF
            column with with the desired data
        '''

        return (self.get_standard_error_measurment_of_a_dataframe(col)*1.96*np.sqrt(2))


        
        

