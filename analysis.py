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
class An():

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
        self.allVariables = ['Date', 'Name']
        self.allVariables.extend(self.variables)

        # fData: pandas dataset
        #   This variable contains the clean version of the data without any extra vsriables or observations
        self.fData = None 

        # grpAthl: pands group object
        #   datasets grouped by athletes
        self.fDataAthl = None

        # grpFall: pands group object
        #   datasets grouped by athletes in the fall
        self.grpFall = None

        # grpWinter: pands group object
        #   datasets grouped by athletes in the Winter
        self.grpWinter = None

        # grpWinter: pands group object
        #   datasets grouped by athletes in the Spring
        self.grpSpring = None       

        # fall: string
        #   contains day and month for the beggining fall season month/day format
        self.fall = fall

        # winter: string
        #   contains day and month for the beggining winter season month/day format
        self.winter = winter

        # spring: string
        #   contains day and month for the beggining spring season month/day format
        self.spring = spring

        # summer: string
        #   contains day and month for the beggining summer season month/day format
        self.summer = summer

        # fallData
        #   This variable contains the observations of the filreted dataset for the Fall season
        self.fallData = None

        # wintData
        #   This variable contains the observations of the filreted dataset for the Winter season
        self.wintData = None

        # spriData
        #   This variable contains the observations of the filreted dataset for the Spring season
        self.spriData = None

        # numObs: int. 
        #   This variable contains the number of observaions
        self.numObs = None

        # numVar: int. 
        #   This variable contains the number of variables
        self.numVar = None

        # des: arr
        #   This variable conatians an array with descriptive statistics for quantitative variables
        self.des = None

        # mean: arr
        #   This variable contains an array with the mean for the number of quantitative variables
        self.mean = None

        # std: arr
        #   This variable contains an array with the Standard Deviation for the number of quantitative variables
        self.std = None

        if path is not None:
            pass

    #set methods
    def set_path(self, path):
        """Replaces path instance variable with `path`.
        NOTE: you have to initialize the class to update

        Parameters:
        -----------
        path: string
        """

        self.path = path

    def set_fall(self, fall):
        """Replaces fall instance variable with `fall`.
        NOTE: you have to initialize the class to update

        Parameters:
        -----------
        fall: string
        """

        self.fall = fall

    def set_winter(self, winter):
        """Replaces winter instance variable with `winter`.
        NOTE: you have to initialize the class to update

        Parameters:
        -----------
        winter: string
        """

        self.winter = winter

    def set_spring(self, spring):
        """Replaces spring instance variable with `spring`.
        NOTE: you have to initialize the class to update

        Parameters:
        -----------
        spring: string
        """

        self.spring = spring

    def set_summer(self, summer):
        """Replaces summer instance variable with `summer`.
        NOTE: you have to initialize the class to update

        Parameters:
        -----------
        summer: string
        """

        self.summer = summer

    #initialize method
    def initialize (self):
        """Initialize the class by cleaning the data and start updating the variables for the class
        """

        #loads the data set into a pandas dataframe
        self.data = pd.read_csv('data/WS.csv')

        # Convert the Date to datetime64
        self.data['Date'] = pd.to_datetime(self.data['Date']) #format='%m/%d/%Y'

        #divides the datasets into the different seasons during the school year
        self.fall= pd.to_datetime(self.fall+'/'+str(self.year))
        self.winter= pd.to_datetime(self.winter+'/'+str(self.year))
        self.spring= pd.to_datetime(self.spring+'/'+str(self.year+1))
        self.summer= pd.to_datetime(self.summer+'/'+str(self.year+1))

        #calls the method to clean the data set
        self.cleanDS()

        #sorts dataset by date
        self.fData=self.fData.sort_values(by='Date',ascending=True)

        #divides the dataset into seasons
        self.fallData = self.filDate(self.fall, self.winter, self.fData)
        self.wintData = self.filDate(self.winter, self.spring, self.fData)
        self.spriData = self.filDate(self.spring, self.summer, self.fData)

        #groups the data for individual atheltes trough the acedemic year
        self.fDataAthl = self.grpAthl(self.fData)

        #groups the data for individual athletes in each season
        self.grpFall= self.grpAthl(self.fallData)
        self.grpWinter= self.grpAthl(self.wintData)
        self.grpSpring= self.grpAthl(self.spriData)

        #sets the reminder of the variables
        self.numObs,self.numVar = self.fData.shape
        self.des = self.fData.describe()
        self.mean = self.fData.mean()
        self.std = self.fData.std()

        return

    #helper methods for initiallize methods
    def grpAthl (self, ds):
        '''Creates an array with individual datasets grouped by athlete
        Parameters:
        -----------
        ds: pandas dataset

        Returns:
        -----------
        pandas group obj
        '''
        
        return ds.groupby('Name')
        
    def cleanDS (self):
        '''Cleans the dataset so that we only have the data from the year we need as well as the observations
            we need. This is saved to self.fData
        Parameters:
        -----------
        
        Returns:
        -----------
        array
        '''

        #filters the columns to the ones we are going to use
        self.fData = self.data[self.allVariables]

        #keeps only the data from that academic year
        self.fData = self.filDate(self.fall, self.summer, self.fData)

        #drops all nan values
        self.fData.dropna()

        return self.fData

    def filDate (self, beggining, end, data):
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

    def initTest(self):
        """test if the we have all the variables we need to start the object

        Returns:
        bool
        """
        pass

    #graphs methods
    #year round
    def varGraphs (self, ds, title):
        '''Creates the graphs of every variable
        Parameters:
        -----------
        ds: pandas dataset
        '''

        dates = ds.loc[:,'Date']
        #subplots
        fig, axs = plt.subplots(len(self.variables), 1, figsize=(20, 20))
        plt.subplots_adjust(hspace=0.8)

        for i in range(len(self.variables)):

            #scatterplot
            axs[i].set_title(self.variables[i])
            axs[i].scatter(dates,ds.loc[:,self.variables[i]])

            #linear regression
            s = ds.set_index('Date')[self.variables[i]]

            y = s
            x = (s.index - pd.Timestamp(0)).days.values
            
            #calculate equation for trendline
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)

            #add trendline to plot
            axs[i].plot(dates, p(x), color="purple", linewidth=3, linestyle="--")
           
        #graph configuration
        plt.suptitle(title, fontsize=20, y=1)
        plt.legend(["Data", "Trend"], loc ="lower right")
        plt.gcf().autofmt_xdate()
        fig.tight_layout()
        plt.show()
        
    def varGraphsMDC (self, ds, title):
        '''Creates the graphs of every variable
        Parameters:
        -----------
        ds: pandas dataset
        '''

        dates = ds.loc[:,'Date']
        #subplots
        fig, axs = plt.subplots(len(self.variables), 1, figsize=(20, 40))
        plt.subplots_adjust(hspace=0.8)

        for i in range(len(self.variables)):

            #get the average of the variable
            avg = ds.loc[:,self.variables[i]].mean()
            mdc = self.getMdcInd(ds.loc[:,self.variables[i]])

            #define the top and bottom bounds
            topMDC = avg + mdc
            botMDC = avg - mdc

            #scatterplot
            axs[i].set_title(self.variables[i])
            axs[i].scatter(dates,ds.loc[:,self.variables[i]])

            #linear regression
            s = ds.set_index('Date')[self.variables[i]]

            y = s
            x = (s.index - pd.Timestamp(0)).days.values
                    
            #calculate equation for trendline
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)

            #add trendline to plot
            axs[i].plot(dates, p(x), color="purple", linewidth=1, linestyle="--")

            #highlight area
            axs[i].fill_between(dates, topMDC, botMDC, color='C0', alpha=0.3)
            axs[i].fill_between(dates, p(x), botMDC, where=(p(x) < botMDC), color='C1', alpha=0.3)
            axs[i].fill_between(dates, topMDC, p(x), where=(p(x) > topMDC), color='C2', alpha=0.3)
            
                
        #graph configuration
        plt.suptitle(title, fontsize=20, y=1)
        plt.legend(["Data", "Trend", "MDC", "Under MDC", "Over MDC"], loc ="lower right")
        plt.gcf().autofmt_xdate()
        fig.tight_layout()
        plt.show()
        
    def varGraphsInd (self, name):
        '''Creates the graphs of every variable for that person
        Parameters:
        -----------
        ds: pandas dataset
        '''
        
        ds = self.fDataAthl.get_group(name)
        self.varGraphs(ds,name)

    def allIndGraphs (self):
        '''Creates the graphs of every variable for that all individuals
        Parameters:
        -----------
        '''
        for name,group in self.fDataAthl:
            self.varGraphsInd(name)

    #seasonal 
    #fall
    def fallGraphs (self):
        '''Creates the graphs of every variables
        Parameters:
        -----------
        '''

        self.varGraphs(self.fallData, 'Fall Team Data')

    def fallGraphsInd (self, name):
        '''Creates the graphs of every variables for an athlete
        Parameters:
        -----------
        name: string
        '''
        ds = self.grpFall.get_group(name)
        self.varGraphs(ds,name)

    def fallGraphsIndAll (self):
        '''Creates the graphs of every variable for that all individuals in the fall season
        Parameters:
        -----------
        '''

        for name,group in self.grpFall:
            self.varGraphsInd(name)

    #winter
    def winterGraphs (self):
        '''Creates the graphs of every variables
        Parameters:
        -----------
        '''

        self.varGraphs(self.wintData, 'Winter Team Data')

    def winterGraphsInd (self, name):
        '''Creates the graphs of every variables for an athlete
        Parameters:
        -----------
        name: string
        '''
        ds = self.grpWinter.get_group(name)
        self.varGraphs(ds,name)

    def winterGraphsIndAll (self):
        '''Creates the graphs of every variable for that all individuals in the fall season
        Parameters:
        -----------
        '''

        for name,group in self.grpWinter:
            self.varGraphsInd(name)

    #spring
    def springGraphs (self):
        '''Creates the graphs of every variables
        Parameters:
        -----------
        '''

        self.varGraphs(self.spriData, 'Spring Team Data')

    def springGraphsInd (self, name):
        '''Creates the graphs of every variables for an athlete
        Parameters:
        -----------
        name: string
        '''
        ds = self.grpSpring.get_group(name)
        self.varGraphs(ds,name)

    def springGraphsIndAll (self):
        '''Creates the graphs of every variable for that all individuals in the spring season
        Parameters:
        -----------
        '''

        for name,group in self.grpSpring:
            self.varGraphsInd(name)

    #team divided by season
    def teamGraphsBySeason(self, title='Data in Different Seasons'):
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
                ds = self.fallData

            if col == 1:
                ds = self.wintData

            if col == 2:
                ds = self.spriData

            for row in range(len(self.variables)):

                dates = ds.loc[:,'Date'] #gets the dates for each data set (X variable)
                
                #scatterplot
                axs[row,col].scatter(dates,ds.loc[:,self.variables[row]])

                #linear regression
                s = ds.set_index('Date')[self.variables[row]]

                y = s
                x = (s.index - pd.Timestamp(0)).days.values
                
                #calculate equation for trendline
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                axs[row,col].set_title('Regression Coefficient: ' + "{:e}".format(z[0])) 

                #add trendline to plot
                axs[row,col].plot(dates, p(x), color="purple", linewidth=3, linestyle="--")
        
        #graph configuration
        plt.suptitle(title, fontsize=20, y=1)
        plt.legend(["Data", "Trend"], loc ="lower right")
        plt.gcf().autofmt_xdate()
        fig.tight_layout()
        plt.show()

    #get Statistics
    #mean
    def getMean(self, variable):
        '''Returns the average for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fData.loc[:,variable].mean()

    def getMeanFall(self, variable):
        '''Returns the average for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fallData.loc[:,variable].mean()

    def getMeanWinter(self, variable):
        '''Returns the average for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.wintData.loc[:,variable].mean()

    def getMeanSpring(self, variable):
        '''Returns the average for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spriData.loc[:,variable].mean()

    #count
    def getCount(self, variable):
        '''Returns the count for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fData.loc[:,variable].count()

    def getCountFall(self, variable):
        '''Returns the count for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fallData.loc[:,variable].count()

    def getCountWinter(self, variable):
        '''Returns the count for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.wintData.loc[:,variable].count()

    def getCountSpring(self, variable):
        '''Returns the count for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spriData.loc[:,variable].count()

    #standard deviation
    def getStd(self, variable):
        '''Returns the standard deviation for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fData.loc[:,variable].std()

    def getStdFall(self, variable):
        '''Returns the standard deviation for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fallData.loc[:,variable].std()

    def getStdWinter(self, variable):
        '''Returns the standard deviation for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.wintData.loc[:,variable].std()

    def getStdSpring(self, variable):
        '''Returns the standard deviation for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spriData.loc[:,variable].std()

    #standard deviation
    def getStd(self, variable):
        '''Returns the standard deviation for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fData.loc[:,variable].std()

    def getStdFall(self, variable):
        '''Returns the standard deviation for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fallData.loc[:,variable].std()

    def getStdWinter(self, variable):
        '''Returns the standard deviation for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.wintData.loc[:,variable].std()

    def getStdSpring(self, variable):
        '''Returns the standard deviation for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spriData.loc[:,variable].std()

    #the minimum value
    def getMin(self, variable):
        '''Returns the min for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fData.loc[:,variable].min()

    def getMinFall(self, variable):
        '''Returns the min for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fallData.loc[:,variable].min()

    def getMinWinter(self, variable):
        '''Returns the min for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.wintData.loc[:,variable].min()

    def getMinSpring(self, variable):
        '''Returns the min for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spriData.loc[:,variable].min()

    #the maximum value
    def getMax(self, variable):
        '''Returns the max for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fData.loc[:,variable].max()

    def getMaxFall(self, variable):
        '''Returns the max for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fallData.loc[:,variable].max()

    def getMaxWinter(self, variable):
        '''Returns the max for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.wintData.loc[:,variable].max()

    def getMaxSpring(self, variable):
        '''Returns the max for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spriData.loc[:,variable].max()

    #the 25th quartile value
    def get25Q(self, variable):
        '''Returns the 25th quartile value for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fData.loc[:,variable].quantile(0.25)

    def get25QFall(self, variable):
        '''Returns the 25th quartile value for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fallData.loc[:,variable].quantile(0.25)

    def get25QWinter(self, variable):
        '''Returns the 25th quartile value for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.wintData.loc[:,variable].quantile(0.25)

    def get25QSpring(self, variable):
        '''Returns the 25th quartile value for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spriData.loc[:,variable].quantile(0.25)

    #the 50th quartile value
    def get50Q(self, variable):
        '''Returns the 50th quartile value for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fData.loc[:,variable].quantile(0.50)

    def get50QFall(self, variable):
        '''Returns the 50th quartile value for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fallData.loc[:,variable].quantile(0.50)

    def get50QWinter(self, variable):
        '''Returns the 50th quartile value for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.wintData.loc[:,variable].quantile(0.50)

    def get50QSpring(self, variable):
        '''Returns the 50th quartile value for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spriData.loc[:,variable].quantile(0.50) 

    #the 75th quartile value
    def get75Q(self, variable):
        '''Returns the 75th quartile value for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fData.loc[:,variable].quantile(0.75)

    def get75QFall(self, variable):
        '''Returns the 75th quartile value for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.fallData.loc[:,variable].quantile(0.75)

    def get75QWinter(self, variable):
        '''Returns the 75th quartile value for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.wintData.loc[:,variable].quantile(0.75)

    def get75QSpring(self, variable):
        '''Returns the 75th quartile value for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return self.spriData.loc[:,variable].quantile(0.75)

    #standard Error of measurement
    def getSem(self, variable):
        '''Returns the SEM value for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.getStd(variable)/np.sqrt(self.getCount(variable)))

    def getSemFall(self, variable):
        '''Returns the SEM value for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.getStdFall(variable)/np.sqrt(self.getCountFall(variable)))

    def getSemWinter(self, variable):
        '''Returns the SEM value for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.getStdWinter(variable)/np.sqrt(self.getCountWinter(variable)))

    def getSemSpring(self, variable):
        '''Returns the SEM value for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.getStdSpring(variable)/np.sqrt(self.getCountSpring(variable)))

    def getSemInd(self, ds):
        '''Returns the SEM value for the stated variable for the dataset
        Parameters
        --------------
        ds: pandas DF
            column with with the desired data
        '''

        return (ds.std()/np.sqrt(ds.count()))

    #minimal detectable change
    def getMdc(self, variable):
        '''Returns the MDC value for the stated variable for the year data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        
        return (self.getSem(variable)*1.96*np.sqrt(2))

    def getMdcFall(self, variable):
        '''Returns the MDC value for the stated variable for the fall data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.getSemFall(variable)*1.96*np.sqrt(2))

    def getMdcWinter(self, variable):
        '''Returns the MDC value for the stated variable for the winter data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.getSemWinter(variable)*1.96*np.sqrt(2))

    def getMdcSpring(self, variable):
        '''Returns the MDC value for the stated variable for the spring data
        Parameters
        --------------
        variable: str 
            name of the variable 
        '''

        return (self.getSemSpring(variable)*1.96*np.sqrt(2))

    def getMdcInd(self, col):
        '''Returns the SEM value for the stated variable for the dataset
        Parameters
        --------------
        col: pandas DF
            column with with the desired data
        '''

        return (self.getSemInd(col)*1.96*np.sqrt(2))


        
        

