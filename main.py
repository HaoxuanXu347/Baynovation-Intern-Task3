# packages
from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

import requests
from bs4 import BeautifulSoup

import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
import plotly.express as px


app = Flask(__name__)

description = {
    "Property": "Identifier for the property.",
    "Current Monthly Payment": "Current monthly payment for the mortgage.",
    "Current Loan Interest Rate (%)": "Current interest rate for the mortgage.",
    "Balance Left on Mortgage ($)": "Outstanding balance on the mortgage.",
    "New Interest Rate (%)": "Proposed new interest rate if the mortgage is refinanced.",
    "Remaining Loan Term (months)": "Remaining term of the current mortgage in months.",
    "New Loan Term (months)": "Proposed new term in months if the mortgage is refinanced.",
    "Closing Cost ($)": "The cost associated with refinancing the mortgage."
}



def lovely_soup(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup

#-------------------------Break Even Point Analysis-----------------------
'''
    This function calculates and provides the break-even point, monthly savings, and new payment details, considering the closing costs and any cash-out during refinancing.
    
    Returns:
    - break_even_point : Number of months it will take to recoup the closing costs from monthly savings.
    - monthly_savings : Difference in monthly payment between the current and refinanced loan.
    - outstanding_principal_balance_adjusted : Adjusted outstanding principal balance after considering the cash-out amount.
    - current_monthly_payment : Monthly payment amount for the current loan.
    - new_monthly_payment : Monthly payment amount for the refinanced loan without considering the cash-out.
    - new_monthly_payment_adjusted : Monthly payment amount for the refinanced loan after considering the cash-out.'''
def mortgage_refinance_calculation(p, r, N, n, new_r, new_n, c, co):
    
    if N is None:
        N = n
    if n is None:
        n = N
    
    principal = p  # Principal amount
    current_interest_rate = r  # Current interest rate
    current_total_term = N  # Current total term
    current_remaining_term = n  # Current remaining term
    new_interest_rate = new_r  # New interest rate
    new_remaining_term = new_n  # New remaining term
    closing_cost = c  # Closing cost
    cash_out = co  # Cash out amount
    # Monthly Payment for Current Mortgage
    current_monthly_interest_rate = current_interest_rate / 12 / 100
    current_monthly_payment = principal * current_monthly_interest_rate * (1 + current_monthly_interest_rate) ** current_total_term / ((1 + current_monthly_interest_rate) ** current_total_term - 1)
    
    # Outstanding Principal Balance
    outstanding_principal_balance = principal * (1 + current_monthly_interest_rate) ** (current_total_term - current_remaining_term) - (current_monthly_payment * ((1 + current_monthly_interest_rate) ** (current_total_term - current_remaining_term) - 1) / current_monthly_interest_rate)
    outstanding_principal_balance_adjusted = outstanding_principal_balance + cash_out
    
    # Monthly Payment for New Mortgage
    # new_monthly_payment: x cash_out
    new_monthly_interest_rate = new_interest_rate / 12 / 100
    new_monthly_payment = outstanding_principal_balance * new_monthly_interest_rate * (1 + new_monthly_interest_rate) ** new_remaining_term / ((1 + new_monthly_interest_rate) ** new_remaining_term - 1)
    new_monthly_payment_adjusted = outstanding_principal_balance_adjusted * new_monthly_interest_rate * (1 + new_monthly_interest_rate) ** new_remaining_term / ((1 + new_monthly_interest_rate) ** new_remaining_term - 1)
    
    # Monthly Payment Savings
    monthly_savings = current_monthly_payment - new_monthly_payment
    
    # Break Even Point
    break_even_point = closing_cost / monthly_savings
    return {
        'break_even_point': break_even_point,
        'monthly_savings': monthly_savings,
        'outstanding_principal_balance_adjusted': outstanding_principal_balance_adjusted, # v cash out
        'current_monthly_payment': current_monthly_payment,
        'new_monthly_payment': new_monthly_payment, # x cash out
        'new_monthly_payment_adjusted': new_monthly_payment_adjusted # v cash out
    }

@app.route('/', methods=['POST'])
def calculate():
    p     = float(request.form['p'])
    r     = float(request.form['r'])
    N     = float(request.form['N'])
    n     = float(request.form['n'])
    new_r = float(request.form['new_r'])
    new_n = float(request.form['new_n'])
    c     = float(request.form['c'])
    co    = float(request.form['co'])

    result =  mortgage_refinance_calculation(p, r, N, n, new_r, new_n, c, co)
    return render_template('mortgage_calculator.html', result=result, description = description)

#-------------------------Cumulative Savings Calculation-----------------------

def get_discount():
    url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202307"
    soup = lovely_soup(url)
    ten_year_rates_elements = soup.find_all('td', headers = "view-field-bc-10year-table-column")
    discount_rate = float(ten_year_rates_elements[-1].text.strip()) if ten_year_rates_elements else None

    return discount_rate

def mortgage_refinance_total_costs_calculation(p, r, N, n, new_r, new_n, c, co, discount_rate):
    
    '''
    Calculate the nominal and present value (PV) costs of maintaining the current mortgage
    versus the costs of refinancing, considering a given discount rate.
    
    Returns:
    A dictionary containing the nominal and PV costs with and without refinancing, 
    and the respective savings.
    
    Note:
    - Assuming the closing cost is paid in advance
    - Doesn't take into account cash out
    '''
    
    # break even point, monthly savings, current monthly payment, new monthly payment
    refinance_results = mortgage_refinance_calculation(p, r, N, n, new_r, new_n, c, co)
    current_monthly_payment = refinance_results['current_monthly_payment']
    new_monthly_payment = refinance_results['new_monthly_payment']
    current_remaining_term = n
    new_remaining_term = new_n
    closing_cost = c
    monthly_discount_rate = (1 + discount_rate) ** (1/12) - 1
    
    # Nominal Total Cost for Current Mortgage
    current_nominal_total_costs = current_monthly_payment * current_remaining_term
    # Nominal Total Cost for New Mortgage
    new_nominal_total_costs = new_monthly_payment * new_remaining_term + closing_cost
    # Nominal Total Cost Savings
    nominal_total_costs_savings = current_nominal_total_costs - new_nominal_total_costs
    
    # PV Total Cost for Current Mortgage
    current_pv_total_costs = sum([current_monthly_payment / ((1 + monthly_discount_rate) ** month) for month in range(1, current_remaining_term + 1)])
    # PV Total Cost for New Mortgage
    new_pv_total_costs = sum([new_monthly_payment / ((1 + monthly_discount_rate) ** month) for month in range(1, new_remaining_term + 1)]) + closing_cost
    # PV Total Cost Savings
    pv_total_costs_savings = current_pv_total_costs - new_pv_total_costs
    
    return{
        'Nominal Cost without Refinance': current_nominal_total_costs,
        'Nominal Cost with Refinance': new_nominal_total_costs,
        'Nominal Savings': nominal_total_costs_savings,
        'PV Cost without Refinance': current_pv_total_costs,
        'PV Cost with Refinance': new_pv_total_costs,
        'PV Savings': pv_total_costs_savings
    }
    

#-------------------------Real-Time Mortgage-----------------------

def get_mortgagerate():
    # The Mortgage News Daily rate index is published daily (weekdays) around 4PM EST
    url = 'https://www.mortgagenewsdaily.com/mortgage-rates'
    soup = lovely_soup(url)
    rate_div= soup.find_all('div', class_ = 'col-sm-4 col-xs-12 rate-product')
    rate_type_div = soup.find_all('div', class_ = 'rate-product-name hidden-xs')
    rates = []
    rate_types = []
    for div in rate_div:
        rate = div.find('div', class_='rate')
        if rate is not None:
            rates.append(rate.text.strip())
    for div in rate_type_div:
        rate_type = div.a.text.strip()
        rate_types.append(rate_type)
    rate_df = pd.DataFrame({'rate_type': rate_types, 'rate': rates})

    return rate_df


#-------------------------Amortization_Schedule-----------------------


@app.route("/")
def home():    
    return render_template("mortgage_calculator.html", description=description)



