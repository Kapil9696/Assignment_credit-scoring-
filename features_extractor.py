import pandas as pd
import json 
from rich.console import Console
from rich.table import Table
from textwrap import shorten
import re

class CreditFeatureExtractor:
    """
    A class for extracting credit features from JSON credit report data.
    """

    def __init__(self, feature_descriptions):
        """Initializes the CreditFeatureExtractor with feature descriptions."""
        self.feature_descriptions = feature_descriptions


    def _preprocess(self, value):
        """
        Handles potential non-numeric values. (Private method)

        Args:
            value: The value to preprocess.

        Returns:
            A float representation of the value, or 0.0 if conversion fails.
        """
        try:
            value = value.split('.')[0]
            to_convert = re.findall(r'\d+', value)
            value = int(''.join(to_convert))
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def extract_features(self, json_data):
        """
        Extracts relevant features from credit report JSON data.

        Args:
            json_data: A list of dictionaries, where each dictionary contains
                       'application_id' and 'data' (the credit report).

        Returns:
            A pandas DataFrame containing the extracted features, or an empty DataFrame
            if the input is invalid.
        """
        if not isinstance(json_data, list):
            print("Error: Input must be a list.")
            return pd.DataFrame()

        features = []

        for entry in json_data:
            if not isinstance(entry, dict) or 'application_id' not in entry or 'data' not in entry:
                print("Warning: Invalid entry format. Skipping.")
                continue

            application_id = entry['application_id']
            credit_report = entry['data']

            try:
                summary = credit_report['consumerfullcredit']['creditaccountsummary']
                agreement_summary = credit_report['consumerfullcredit']['creditagreementsummary']
                employment_history = credit_report['consumerfullcredit']['employmenthistory']

                total_debt = self._preprocess(summary.get('totaloutstandingdebt', 0))
                credit_score = self._preprocess(summary.get('rating', 0))
                amount_arrear = self._preprocess(summary.get('amountarrear', 0))
                total_monthly_installment = self._preprocess(summary.get('totalmonthlyinstalment', 0))
                num_accounts = len(agreement_summary) if agreement_summary else 0

                num_open_accounts = sum(1 for account in agreement_summary if account.get('accountstatus') == 'Open') if agreement_summary else 0
                num_closed_accounts = sum(1 for account in agreement_summary if account.get('accountstatus') == 'Closed') if agreement_summary else 0
                num_performing_accounts = sum(1 for account in agreement_summary if account.get('performancestatus') == 'Performing') if agreement_summary else 0
                num_non_performing_accounts = sum(1 for account in agreement_summary if account.get('performancestatus') != 'Performing') if agreement_summary else 0

                total_credit_limit = sum(self._preprocess(account.get('openingbalanceamt', 0)) for account in agreement_summary) if agreement_summary else 0
                avg_credit_line = total_credit_limit / num_accounts if num_accounts > 0 else 0

                employment_status = employment_history[0].get('occupation') if employment_history else None

            except (KeyError, TypeError, IndexError) as e:
                print(f"Error processing credit report for application ID {application_id}: {e}. Setting default values.")
                total_debt = credit_score = amount_arrear = total_monthly_installment = 0
                num_accounts = num_open_accounts = num_closed_accounts = 0
                num_performing_accounts = num_non_performing_accounts = 0
                avg_credit_line = 0
                employment_status = None

            features.append({
                "application_id": application_id,
                "total_debt": total_debt,
                "credit_score": credit_score,
                "amount_arrear": amount_arrear,
                "total_monthly_installment": total_monthly_installment,
                "num_accounts": num_accounts,
                "num_open_accounts": num_open_accounts,
                "num_closed_accounts": num_closed_accounts,
                "num_performing_accounts": num_performing_accounts,
                "num_non_performing_accounts": num_non_performing_accounts,
                "avg_credit_line": avg_credit_line,
                "employment_status": employment_status
            })

        return pd.DataFrame(features)

    def display_feature_relevance(self, max_description_length=150, padding=1):
        """Displays feature relevance using Rich, with shortened descriptions and padding."""

        console = Console()
        table = Table(title="Feature Relevance to Risk Scoring", padding=(padding, padding))

        table.add_column("Feature Name", style="cyan", width=25)
        table.add_column("Relevance to Risk Scoring", style="magenta", overflow="fold")

        for feature_name, description in self.feature_descriptions.items():
            shortened_description = shorten(description, width=max_description_length, placeholder="...")
            table.add_row(feature_name, shortened_description)

        console.print(table)

feature_descriptions = {
    "total_debt": "Reflects overall indebtedness. Higher debt strongly correlates with increased default risk. A high total debt compared to income indicates financial strain and a higher likelihood of missed payments, making it a critical factor in risk assessment.",
    "credit_score": "A fundamental indicator of creditworthiness derived from credit bureau data. Lower scores directly indicate a higher risk of default, reflecting a history of missed payments, delinquencies, or other negative credit events. This is a primary factor in most credit risk models.",
    "arrear_amount": "Directly indicates financial distress. Higher amounts in arrears are a very strong predictor of default, signaling an immediate inability to meet current payment obligations and a significantly elevated risk profile.",
    "total_monthly_payment": "Represents the borrower's total monthly debt obligations across all credit accounts. A high value relative to income indicates potential financial strain and a higher risk of default, as a larger portion of income is dedicated to debt repayment, leaving less room for unexpected expenses.",
    "num_total_accounts": "The total number of credit accounts held by the borrower, including both open and closed accounts. A higher number can sometimes indicate higher risk, especially when combined with high debt levels, but it can also reflect a diverse and well-managed credit history. It should be considered in conjunction with other factors like payment history and credit utilization.",
    "num_open_accounts": "The number of currently open credit accounts. A large number of open accounts could suggest the borrower is actively taking on more debt, potentially increasing risk in the future. This can indicate a higher potential for future debt accumulation and should be analyzed in relation to income and existing debt levels.",
    "num_closed_accounts": "The number of closed credit accounts. Can provide insights into the borrower's past credit behavior and financial management. A large number of closed accounts might indicate past financial difficulties, strategic debt management, or simply the natural closure of accounts over time. This historical context is valuable in risk assessment.",
    "num_performing_accounts": "The number of accounts with a performing status, indicating a consistent history of timely payments. A higher number indicates a good track record of repayments and a correspondingly lower assessed risk of future default.",
    "num_non_performing_accounts": "The number of accounts that are not performing (e.g., in default, collections, or with late payments). A critical and highly predictive risk indicator. More non-performing accounts strongly correlate with a substantially higher default risk.",
    "average_credit_line": "The average credit limit across all the borrower's credit accounts. Can be used to assess the borrower's access to credit and their potential for credit utilization. A low average credit line might indicate limited creditworthiness, while a very high average might suggest overextension and increased risk if not managed responsibly. It's important to consider this in relation to income and debt levels.",
    "employment_status": "Indicates the borrower's current employment situation. Stable employment is generally considered a positive factor and reduces risk, while unemployment or unstable employment can significantly increase the assessed risk of default. This should ideally be further analyzed with details like job tenure, industry, and income stability.",
}