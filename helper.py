import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score, f1_score, roc_curve, auc #To evaluate our model
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px
import math
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.metrics import confusion_matrix
import shap
from termcolor import colored


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

def find_optimal_threshold(fpr_cat, tpr_cat):
    """
    Find the optimal threshold based on the minimum distance to the top-left corner of the ROC space.

    Parameters:
        fpr_cat (list or array): False Positive Rates.
        tpr_cat (list or array): True Positive Rates.

    Returns:
        dict: A dictionary with the optimal threshold's False Positive Rate, True Positive Rate, and Distance.
    """
    # Input validation
    if len(fpr_cat) != len(tpr_cat):
        raise ValueError("The lengths of fpr_cat and tpr_cat must be equal.")
    if not (all(isinstance(i, (int, float)) for i in fpr_cat) and all(isinstance(i, (int, float)) for i in tpr_cat)):
        raise ValueError("All elements in fpr_cat and tpr_cat must be numerical values.")
    
    data = {'fpr_cat': fpr_cat, 'tpr_cat': tpr_cat}
    optimal_df = pd.DataFrame(data)

    optimal_df['Distance'] = np.sqrt((optimal_df['fpr_cat'] - 0) ** 2 + (optimal_df['tpr_cat'] - 1) ** 2)

    optimal_index = optimal_df['Distance'].idxmin()

    # Get the optimal threshold
    optimal_threshold = optimal_df.iloc[optimal_index]

    return optimal_threshold


def format_percent(value):
     return f'{value:.2%}'


def evaluate_model(model, X_test, y_test):
    """
    Generate comprehensive model evaluation.

    Parameters:
        model: Trained model to be evaluated.
        X_test: Test features.
        y_test: True labels for the test set.

    Returns:
        dict: A dictionary containing various evaluation metrics.
    """
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        beta_score = fbeta_score(y_test, y_pred, beta=2)
        f1score = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        classification_rpt = classification_report(y_test, y_pred)
        
        sensitivity_recall = cm[0, 0] / (cm[0, 0] + cm[1, 0])
        specificity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr_cat, tpr_cat, thresholds = roc_curve(y_test, y_pred_prob)
        auc_score = auc(fpr_cat, tpr_cat)
        
        optimal_threshold = find_optimal_threshold(fpr_cat, tpr_cat)

        print(classification_rpt)

        print(colored('AUC : ', 'red', attrs=['bold']), colored(format_percent(auc_score), 'red', attrs=['bold']))

        print(colored('Sensitivity_recall : ', 'blue', attrs=['bold']), colored(format_percent(sensitivity_recall), 'blue', attrs=['bold']))

        print(colored('Specificity: ', 'green', attrs=['bold']), colored(format_percent(specificity), 'green', attrs=['bold']))

        print(colored('Precision: ', 'yellow', attrs=['bold']), colored(format_percent(precision), 'yellow', attrs=['bold']))

        print(colored('Accuracy: ', 'magenta', attrs=['bold']), colored(format_percent(accuracy), 'magenta', attrs=['bold']))

        

        # Plot ROC curve
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_cat, tpr_cat)
        plt.plot(optimal_threshold['fpr_cat'], optimal_threshold['tpr_cat'], 
                 'ro', label='Optimal Point', markersize=10)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.savefig('roc_curve.png')
        plt.show()

        return {
            'accuracy': accuracy,
            'beta_score': beta_score,
            'f1_score': f1score,
            'sensitivity_recall': sensitivity_recall,
            'specificity': specificity,
            'precision': precision,
            'auc_score': auc_score,
            'optimal_threshold': optimal_threshold
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def feature_encoding(df, categorical_columns, numerical_columns):
    """
    Encode categorical features and scale numerical features.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        categorical_columns (list): List of columns to be one-hot encoded.
        numerical_columns (list): List of columns to be scaled.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features and scaled numerical features.
    """
    try:
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input data must be a pandas DataFrame.")
        if not all(isinstance(col, str) for col in categorical_columns):
            raise ValueError("Categorical columns must be provided as a list of strings.")
        if not all(isinstance(col, str) for col in numerical_columns):
            raise ValueError("Numerical columns must be provided as a list of strings.")
        
        # Initialize OneHotEncoder and StandardScaler
        encoder = OneHotEncoder(sparse_output=False)
        scaler = StandardScaler()

        # Fit and transform the categorical columns
        one_hot_encoded = encoder.fit_transform(df[categorical_columns])

        # Create a DataFrame with the encoded columns
        one_hot_df = pd.DataFrame(one_hot_encoded, 
                                  columns=encoder.get_feature_names_out(categorical_columns))

        # Scale the numerical columns
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

        # Concatenate the original df (without categorical columns) with the encoded columns
        df = pd.concat([df.drop(categorical_columns, axis=1), one_hot_df], axis=1)

        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def train_model(X_train, y_train, X_test, y_test):
    """
    Train and validate a CatBoost model with cross-validation using Bayesian optimization.

    Parameters:
        X_train (pd.DataFrame or np.array): Training feature set.
        y_train (pd.Series or np.array): Training labels.
        X_test (pd.DataFrame or np.array): Test feature set.
        y_test (pd.Series or np.array): Test labels.

    Returns:
        model: The trained CatBoost model with the best found parameters.
        best_params (dict): Best parameters found during the optimization.
        score (float): Score of the best model on the test set.
    """
    try:
        param_space = {
            'iterations': (300, 1000),  # Number of boosting iterations
            'learning_rate': (0.01, 0.3),  # Learning rate
            'depth': (4, 12),  # Tree depth
            'l2_leaf_reg': (1, 10),  # L2 regularization term for leaf values
            'random_seed': [42]  # Random seed for reproducibility
        }
        
        catboost = CatBoostClassifier()

        # Initialize BayesSearchCV
        bayes_search = BayesSearchCV(
            catboost,
            param_space,
            n_iter=50,  # Number of parameter settings that will be sampled
            cv=5,  # Number of cross-validation folds
            random_state=42,
            verbose=1,
            n_jobs=-1  # Number of parallel jobs to run
        )

        bayes_search.fit(X_train, y_train)

        # Evaluate the best model
        model = bayes_search.best_estimator_
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        return model, bayes_search.best_params_, score
    
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return None, None, None


def load_catboost_model(model_path):
    """
    Load a CatBoost model from a file.

    Parameters:
        model_path (str): The path to the saved model file.

    Returns:
        model: The loaded CatBoost model.
    """
    try:
        # Load the model
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None


def prepare_data_pipeline(df, target_variable='default'):
    """
    Prepare data for model training by applying SMOTETomek for balancing and splitting into train and test sets.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        target_variable (str): The name of the target variable column. Default is 'default'.

    Returns:
        X_train, X_test, y_train, y_test: Splits of the data ready for training and testing.
    """
    try:
        y = df[target_variable]
        X = df.drop(target_variable, axis=1)

        smt = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smt.fit_resample(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        return None, None, None, None

def summarize_feature_impact(shap_values, feature_importance_df):
    """
    Summarize the impact of features based on SHAP values and feature importance.

    Parameters:
        shap_values (shap.Explanation): SHAP values for the model predictions.
        feature_importance_df (pd.DataFrame): DataFrame containing feature importances.

    Returns:
        summary (str): A summary of the key features and their impact.
    """
    try:
        # Get the mean absolute SHAP values for each feature
        shap_mean_abs = np.abs(shap_values.values).mean(axis=0)
        shap_importance = pd.DataFrame(list(zip(shap_values.feature_names, shap_mean_abs)), columns=['feature', 'mean_abs_shap'])
        shap_importance = shap_importance.sort_values(by='mean_abs_shap', ascending=False)

        # Combine SHAP importance with feature importance
        combined_importance = feature_importance_df.set_index('Feature Id').join(shap_importance.set_index('feature'))
        combined_importance = combined_importance.sort_values(by='mean_abs_shap', ascending=False)

        # Create a summary
        summary = "### Feature Impact Summary ###\n"
        summary += "\nTop Features based on SHAP values:\n"
        for idx, row in shap_importance.head(10).iterrows():
            summary += f"Feature: {row['feature']}, Mean Absolute SHAP Value: {row['mean_abs_shap']:.4f}\n"

        summary += "\nTop Features based on Model's Feature Importance:\n"
        for idx, row in feature_importance_df.head(10).iterrows():
            summary += f"Feature: {row['Feature Id']}, Importance: {row['Importances']:.4f}\n"

        summary += "\nCombined Analysis:\n"
        for idx, row in combined_importance.head(10).iterrows():
            summary += f"Feature: {idx}, Importance: {row['Importances']:.4f}, Mean Absolute SHAP Value: {row['mean_abs_shap']:.4f}\n"

        return summary
    except Exception as e:
        return f"An error occurred while summarizing feature impact: {e}"


def analyze_business_impact(y_true, X, model, avg_loan=10000):
    """
    Estimate financial impact of model implementation.

    Parameters:
        y_true (pd.Series or np.array): True labels for the test set.
        X (pd.DataFrame or np.array): Feature set used for predictions.
        model (object): Trained model to be evaluated.
        avg_loan (int, optional): Average loan amount. Default is $10,000.

    Returns:
        pd.DataFrame: DataFrame containing the financial impact at various thresholds.
    """
    try:
       # Feature importance plot
        feature_importance = model.get_feature_importance(prettified=True)
        feature_importance_df = pd.DataFrame(feature_importance)
        
        # Ensure feature_importance_df has the correct columns
        if 'Importances' in feature_importance_df.columns and 'Feature Id' in feature_importance_df.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importances", y="Feature Id", data=feature_importance_df)
            plt.title('Feature Importances')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.show()
        else:
            print("Feature importance DataFrame does not have the expected columns.")

        # Predicted probabilities
        y_pred_proba = model.predict_proba(X)[:, 1]

        # SHAP values
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X)

        thresholds = np.linspace(0, 1, 100)
        results = []

        for thresh in thresholds:
            y_pred = y_pred_proba > thresh
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Calculate financial impact
            losses_prevented = tp * avg_loan
            false_positive_cost = fp * avg_loan * 0.2
            net_benefit = losses_prevented - false_positive_cost

            results.append({
                'threshold': thresh,
                'net_benefit': net_benefit,
                'losses_prevented': losses_prevented,
                'false_positive_cost': false_positive_cost
            })

        impact_df = pd.DataFrame(results)
        optimal = impact_df.loc[impact_df['net_benefit'].idxmax()]

         
        print(f"Optimal threshold: {optimal['threshold']:.2f}")
        print(f"Maximum net benefit: €{optimal['net_benefit']:,.0f}")
        print(f"Estimated defaults prevented: {optimal['losses_prevented']/avg_loan:.0f}")
        print(f"Associated opportunity cost: €{optimal['false_positive_cost']:,.0f}")
        
        summary= summarize_feature_impact(shap_values, feature_importance_df)
        print(summary)

        return impact_df, optimal

    except Exception as e:
        print(f"An error occurred during business impact analysis: {e}")
        return None, None




