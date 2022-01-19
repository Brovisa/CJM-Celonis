# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 07:59:57 2019
GG Analytics to Celonis Eventlog
@author: YoupSuurmeijer
"""
#%% Load libraries

import argparse
from googleapiclient.discovery import build
import httplib2
from oauth2client import client
from oauth2client import file
from oauth2client import tools
import tkinter as tk
from tkinter import filedialog, ttk, Message
import pandas as pd
from pandas import json_normalize
from faker import Faker
from datetime import datetime
import os
import time

#%% Initialize variables
Faker.seed(12345)

#Following variables need to be adjusted for website settings
USER_EXPLORER_DATA = "Analytics User Explorer 20210201-20210228.csv" #Name of user IDs file
CLIENT_SECRECTS_NAME = "client_secret_eventlogAPI.json" #Name of json key-value pair file
CLIENT_SECRETS_PATH = os.getcwd() + "/" + CLIENT_SECRECTS_NAME # Path to client_secrets.json file, by default it looks in python file location
USER_EXPLORER_DATA_PATH = os.getcwd() + "/" + USER_EXPLORER_DATA # Path to client_secrets.json file, by default it looks in python file location
VIEW_ID = 'ga:179138379' #View ID for GG analytics data to query (see reference guide)
DATE_START = "2021-02-01" #Set date limits for the query function ensure these match with downloaded client IDs !!!
DATE_END = "2021-02-28"
TEST = False
SPEED_LIMIT = 1000 #Maximum amount of requests per 100 seconds, useful when reaching API request quota
PRINT_SPEED = True #Boolean variable indicating whether to print the speed of the API calls to the console


#Leave following variables as is unless there is a specific reason to adjust
SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
DISCOVERY_URI = ('https://analyticsreporting.googleapis.com/$discovery/rest')
COLS = ['sessions', 'activities']
SKIP_ROWS = 6

#%% Object definitions

'''
class progressBar():
    """UI progress bar object """
    def __init__(self, N):
        self.master = tk.Tk()
        self.N = N
        self.min = 0
        self.max = 100
        self.step = self.max/self.N
        self.progressbar = ttk.Progressbar(self.master, orient = "horizontal", length = 300, mode = "determinate")
        self.progressbar.pack(side=tk.TOP)
        self.progressbar["value"] = self.min
        self.progressbar["maximum"] = self.max
    
    def update(self):
        self.progressbar["value"] += self.step 
        self.progressbar.update()

    def destroy(self):
        message = Message(self.master, text="this is a message")
        message.pack()
        self.master.withdraw()
 '''

#%% Function definitions
def get_user_ids(col):
    """Initialize file dialog to select csv with user ID's and return object with UIDs
    Inputs:
    col (string): column name of column in csv file which contains the User ID's
    
    Returns:
    pd series object with the user ID's to be used in querying GG analytics
    """
    root = tk.Tk()
    FILE_PATH = False
    
    while not FILE_PATH:
        FILE_PATH = filedialog.askopenfilename()
    
    if FILE_PATH:
        root.withdraw()
        temp = pd.read_csv(FILE_PATH, skiprows = SKIP_ROWS, dtype = {col : str})
    
    return(temp[col])


def get_user_ids_from_csv(col):
    """Select csv with user ID's and return object with UIDs
    Inputs:
    col (string): column name of column in csv file which contains the User ID's

    Returns:
    pd series object with the user ID's to be used in querying GG analytics
    """

    if os.path.isfile(USER_EXPLORER_DATA_PATH):
        temp = pd.read_csv(USER_EXPLORER_DATA_PATH, skiprows=SKIP_ROWS, dtype={col: str})

    return (temp[col])

    

def initialize_analyticsreporting():
  """Initializes the analyticsreporting service object.

  Returns:
    analytics an authorized analyticsreporting service object.
  """
  # Parse command-line arguments.
  parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      parents=[tools.argparser])
  flags = parser.parse_args([])

  # Set up a Flow object to be used if we need to authenticate.
  flow = client.flow_from_clientsecrets(
      CLIENT_SECRETS_PATH, scope=SCOPES,
      message=tools.message_if_missing(CLIENT_SECRETS_PATH))

  # Prepare credentials, and authorize HTTP object with them.
  # If the credentials don't exist or are invalid run through the native client
  # flow. The Storage object will ensure that if successful the good
  # credentials will get written back to a file.
  storage = file.Storage('analyticsreporting.dat')
  credentials = storage.get()
  if credentials is None or credentials.invalid:
    credentials = tools.run_flow(flow, storage, flags)
  http = credentials.authorize(http=httplib2.Http())

  # Build the service object.
  analytics = build('analyticsreporting', 'v4', http=http, discoveryServiceUrl=DISCOVERY_URI)

  return analytics

def get_user_data(analytics, user_id):
    """Initialize file dialog to select csv with user ID's and return object with UIDs
    Inputs:
    analytics (object): GG analytics connection API instance
    user_id (string): user ID to query from GG analytics API
    
    Returns:
    JSON formatted object with API return values
    """
    return analytics.userActivity().search(
            body = {
            "viewId": VIEW_ID,
            "user": {
                "type": "CLIENT_ID",
                "userId": user_id
                },
            "dateRange": {
                "startDate": DATE_START,
                "endDate": DATE_END,
                }
            }
        ).execute()
            

def unnest(df, col):
    """Function to unnest nested dataframe list type objects
    Inputs:
    df (pd.DataFrame): dataframe with nested lists
    col: column name of column containing nested list objects
    Returns:
    pd dataframe object with the unnested items
    """
    unnested = (df.apply(lambda x: pd.Series(x[col]), axis=1)
                .stack()
                .reset_index(level=1, drop=True))
    unnested.name = col
    return df.drop(col, axis=1).join(unnested)

def json_to_df(json, cols, user_id):
    """Function to convert the GG analytics return object to a pd dataframe object
    Inputs:
    json (JSON): GG analytics API return object
    cols (List): List of nested column names
    Returns:
    pd dataframe object containing all data from the GG analytics return object
    """
    #Unnest the sessions from the raw json response
    sessions_temp = pd.concat((pd.DataFrame(i) for i in json[cols[0]]), axis = 0)
    #Unnest the activities using json normalizer
    activities_temp = json_normalize(sessions_temp[cols[1]])
    #Define output dataframe
    output = pd.concat([sessions_temp.reset_index(drop=True), activities_temp.reset_index(drop=True)], axis=1)
    #Add user ID as column
    output['userId'] = user_id
    
    #Drop unneccessary columns
    if 'customDimension' in output:
        output = output.drop(['activities', 'customDimension'] , axis = 1)
    else:
        output = output.drop(['activities'] , axis = 1)
    
    
    return(output)

def add_functional_name(df, name_col):
    """ Simple function to add random human names to dataframe instead of GG analytics UIDs """
    N = df[name_col].unique().size
    names = []
    for _ in range(2*N):
        names.append(Faker().name())
    
    names = pd.Series(names)
    names = names.unique()[:N]
    
    if names.size == N: 
        df['userName (fictional)'] = pd.Categorical(df[name_col])
        df['userName (fictional)'] = df['userName (fictional)'].cat.rename_categories(names)
        return(df)
    else:
        return(df)

def export_to_csv(df):
    """ Export eventlog dataframe to csv """
    path = os.getcwd() + "/output"
    
    if not os.path.isdir(path):
        os.mkdir(path)
        
    output_name = path + "/eventlog (" + str(datetime.today().strftime('%Y-%m-%d')) + ").csv"
    
    unique = False
    n = 0
    
    while not unique:
        n += 1
        if not os.path.isfile(output_name):
            unique = True
        else:
            output_name = output_name.split(".csv")[0] + " (" + str(n) + ").csv"
    
    df.to_csv(output_name)     

def main():
    #Get the list with all user IDs (from manual download)
    user_ids = get_user_ids_from_csv('Client ID') # was get_user_ids('Client ID')
    #If TEST = True limit the amount of user ID's to the first 3
    if TEST:
        user_ids = user_ids[:3]
    #initialize output dataframe
    output = pd.DataFrame()
    #Initialize the API connection
    analytics = initialize_analyticsreporting()
    ## Initialize progress bar
    # progress_bar = progressBar(user_ids.size)
    #Loop through all user ID's and request API response
    for uid in user_ids:
        start_time = time.time()
        print("Retrieving user info for: ", uid, " (", user_ids[user_ids == uid].index[0] + 1 ,"/", user_ids.size, ")")
        # progress_bar.update()
        response = get_user_data(analytics, user_id = uid)

        df_temp = json_to_df(response, cols = COLS, user_id = uid)
        output = pd.concat([output.reset_index(drop=True), df_temp.reset_index(drop=True)], axis=0)
        #Check if the API calls are going too fast, and if so, delay with appropriate amount of seconds
        time_passed = time.time()-start_time
        if time_passed < 100/SPEED_LIMIT:
            deltat =  100/SPEED_LIMIT - time_passed 
            time.sleep(deltat)
        #Print current speed of API calls if True
        if PRINT_SPEED:
            print("Current speed = ", round(1/(time.time()-start_time),2), "API calls per second")
        
    
    output = add_functional_name(output, 'userId')
    export_to_csv(output)
#    progress_bar.destroy()
    return(output)



if __name__ == '__main__':
    global data
    data = main()