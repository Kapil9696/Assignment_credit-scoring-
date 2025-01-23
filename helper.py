import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px
import math


# Function to convert to months
def to_months(residence_time):
    try:
        num, unit = residence_time.split()
        num = int(num)        
        if 'year' in unit:
            return num * 12
        return num
    except:
        num = int(0) 



# Define the new category mappings
purpose_category_mapping = {
    'radio/tv': 'Entertainment Purpose',
    'car (new)': 'Transportation Purpose',
    'furniture': 'Household Needs Purpose',
    'car (used)': 'Transportation Purpose',
    'business': 'Career_Educational Purpose',
    'education': 'Career_Educational Purpose',
    'repairs': 'Household Needs Purpose',
    'domestic appliances': 'Household Needs Purpose',
    'others': 'Others',
    'retraining': 'Career_Educational Purpose'
}


# Define bins and labels
bins = [0, 100, 500, 1000, 5000, 30000]
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
savings_balance_category = [bins,labels]

#employment_length_category
bins = [0, 12, 60, 120, 180,float('inf')]
labels = ['0-1 years', '2-5 years', '6-10 years', '11-15 years','15+ years']
employment_length_category = [bins,labels]


def numerical_varaible_analysis(df,numerical):
    #= ['checking_balance','months_loan_duration' ,'amount', 'age','residence_history']
    n_cols = 2  # Number of columns in the grid
    n_rows = math.ceil(len(numerical) * 2 / n_cols)  # Calculate rows based on number of features
    
    plt.figure(figsize=(15, n_rows * 5))
    i = 1
    
    for col in numerical:
        # Plot histogram
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(x=col, data=df, hue='default', kde=True)
        plt.title(f"{col} - Histogram", size=16)
        plt.xlabel("")
        plt.ylabel("Count", size=14)
        i += 1
    
        # Plot boxplot
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(x='default', y=col, data=df)
        plt.title(f"{col} - Boxplot", size=16)
        plt.xlabel("Default", size=14)
        plt.ylabel(col, size=14)
        i += 1
    
    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    plt.show()

def categorical_variables_analysis(df,categorical):
    
    # categorical = [col for col in df.columns if col not in numerical]

    plt.figure(figsize=(20, 35))
    for i, col in enumerate(df[categorical], 1):
    
        
        # Plot corresponding variable
        plt.subplot(6, 3, i)
        ax = sns.countplot(x=col, data=df,hue='Target')
        
        # Adjust labels and title. Display each class count
        plt.xlabel("")
        plt.ylabel("Count", size=14)
        plt.title(col.title(), size=16)
        #ax.bar_label(ax.containers[0])
        # Add counts on the bars
        for container in ax.containers:
            ax.bar_label(container)                   
        
                           
        
        # # Rotate long class names in some columns to make them easier to read
        if col in ['credit_history', 'purpose', 'job']:
            plt.xticks(rotation=45, horizontalalignment='right')
            
    plt.subplots_adjust(hspace=0.7, wspace=0.25)




