"""
Project Name: Refinance Break Even Point Web Application
Author: Haoxuan Xu
Credit: Shuyan Kang
Date: 8/9/23 - 8/22/23

Description:
    develop a web application or GUI that assists users in calculating the refinance break-even point for their loans. 
    The main functionality of this application will involve allowing users to input relevant information such as current 
    loan details, interest rates, loan term, closing costs, and any potential new loan details.

Functions:
    1. Break Even Point Analysis
    2. Cumulative Savings Calculation
    3. Mortgage Rates
    4. Amortization_Schedule
    
"""

# packages
from flask import Flask, render_template, request, redirect
import os
import requests
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import ipywidgets as widgets
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.display import clear_output
from matplotlib.ticker import FuncFormatter

matplotlib.use('Agg')

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

def thousands_formatter(x, pos):
    return f'{x/1000:.0f}K'

def save_plot(plt, filename):
    path = f"static/{filename}"
    plt.savefig(path)
    return path


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

def loan_amortization_schedule(principal, rate, term):
    
    '''
    Generate an amortization schedule for a loan given the principal amount, interest rate, and term.

    Returns:
    A DataFrame containing the amortization schedule, monthly outstanding principal,
    monthly payments, principal and interest components of the payment, 
    and cumulative payments over the term of the loan.
    '''
    
    monthly_interest_rate = rate / 12 / 100
    monthly_payment = principal * monthly_interest_rate * (1 + monthly_interest_rate) ** term / ((1 + monthly_interest_rate) ** term - 1)
    
    outstanding_principal = principal
    records = []
    cumulative_principal_payment = 0
    cumulative_interest_payment = 0 
    
    for month in range(1, term + 1):
        interest_payment = outstanding_principal * monthly_interest_rate
        principal_payment = monthly_payment - interest_payment
        cumulative_principal_payment += principal_payment
        cumulative_interest_payment += interest_payment
        cumulative_payment = cumulative_principal_payment + cumulative_interest_payment
        outstanding_principal -= principal_payment
        records.append([month, outstanding_principal, monthly_payment, principal_payment, interest_payment, 
                        cumulative_principal_payment, cumulative_interest_payment, cumulative_payment])
    df = pd.DataFrame(records, columns=['Month', 'Outstanding Principal', 'Monthly Payment', 'Principal Payment', 'Interest Payment', 
                                        'Cumulative Principal Payment', 'Cumulative Interest Payment', 'Cumulative Payment'])
    return df



#---------------------------------Visualization---------------------------------
def plot_monthly_payment(p, r, N, n, new_r, new_n, c, co):
    
    '''
    Plot a comparison of monthly payments between the current mortgage plan and the potential refinanced mortgage plans.
    '''
    
    # Results from the mortgage_refinance_calculation function
    refinance_results = mortgage_refinance_calculation(p, r, N, n, new_r, new_n, c, co)
    current_monthly_payment = refinance_results['current_monthly_payment']
    new_monthly_payment = refinance_results['new_monthly_payment']
    new_monthly_payment_adjusted = refinance_results['new_monthly_payment_adjusted']
    outstanding_payment = refinance_results['outstanding_principal_balance_adjusted'] - co
    outstanding_payment_adjusted = refinance_results['outstanding_principal_balance_adjusted']
    monthly_savings = refinance_results['monthly_savings']
    monthly_savings_adjusted = current_monthly_payment - new_monthly_payment_adjusted
    
    # The types of payments and their labels
    payments_type = [current_monthly_payment, new_monthly_payment]
    labels = [
        f'Current Mortgage Plan\nOutstanding: ${outstanding_payment:,.2f}\n{r}% APR\n{n} months remaining',
        f'Refinance Mortgage Plan Without Cash Out\nOutstanding: ${outstanding_payment:,.2f}\n{new_r}% APR\n{new_n} months remaining'
    ]
    
    # Include cash-out details if provided
    if co != 0:
        payments_type.append(new_monthly_payment_adjusted)
        labels.append(f'Refinance Mortgage Plan With Cash Out\nOutstanding: ${outstanding_payment_adjusted:,.2f}\n{new_r}% APR\n{new_n} months remaining')
    
    plt.figure(figsize = (4, 3))
    bars = plt.bar(labels, payments_type, color = ['black', 'blue', 'lightblue'])
    plt.ylabel('Monthly Payment ($)')
    plt.title('Monthly Payment Comparison')
    plt.xticks(ha='center', fontsize = 8)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'${bar.get_height():,.2f}', ha = 'center', va = 'bottom', fontsize = 15)
        
    plt.axhline(y = new_monthly_payment, color = 'red', linestyle = '--')
    plt.axhline(y = new_monthly_payment_adjusted, color = 'green', linestyle = '--')
    
    # Annotate some details
    plt.annotate(f'Monthly Savings: ${monthly_savings:,.2f}', 
                 xy = (1, new_monthly_payment), 
                 xycoords = 'data', 
                 xytext = (50, 20), 
                 textcoords = 'offset points', 
                 arrowprops = dict(arrowstyle = "->", lw = 1.5))
    
    plt.annotate(f'Difference: ${monthly_savings_adjusted:,.2f}', 
                 xy = (2, new_monthly_payment_adjusted), 
                 xycoords = 'data', 
                 xytext = (50, 10), 
                 textcoords = 'offset points', 
                 arrowprops = dict(arrowstyle = "->", lw = 1.5))
    
    plt.tight_layout()
    
    plot_path = os.path.join('static', 'monthly_payment.png')
    plt.savefig(plot_path)
    
    return plot_path

def plot_loan_amortization_schedule(p, r, N, n, new_r, new_n, c, co):
    
    '''
    Plot a comparison of the cumulative payments over time for the current loan 
    versus a potential refinanced loan.
    '''
    
    refinance_results = mortgage_refinance_calculation(p, r, N, n, new_r, new_n, c, co)
    outstanding_principal_balance = refinance_results['outstanding_principal_balance_adjusted'] - co
    
    # Generate amortization schedules
    current_schedule_df = loan_amortization_schedule(outstanding_principal_balance, r, n)
    new_schedule_df = loan_amortization_schedule(outstanding_principal_balance, new_r, new_n)
    # Add a year column
    current_schedule_df['Year'] = ((current_schedule_df['Month'] - 1) / 12).astype(int) + 1
    new_schedule_df['Year'] = ((new_schedule_df['Month'] - 1) / 12).astype(int) + 1
    # Group by year and sum the payments
    current_grouped = current_schedule_df.groupby('Year').max()
    new_grouped = new_schedule_df.groupby('Year').max()
    
    plt.figure(figsize = (14,7))

    plt.scatter(current_grouped.index, current_grouped['Cumulative Payment'], label = 'Current Cumulative Payment', color = 'black')
    plt.scatter(new_grouped.index, new_grouped['Cumulative Payment'], label = 'New Cumulative Payment', color = 'blue')
    plt.plot(current_grouped.index, current_grouped['Cumulative Payment'], color = 'black')
    plt.plot(new_grouped.index, new_grouped['Cumulative Payment'], color = 'blue')

    plt.title('Cumulative Payments Over Time')
    plt.xlabel('Year')
    plt.ylabel('Amount ($)')
    formatter = FuncFormatter(thousands_formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join('static', 'loan_schedule.png')
    plt.savefig(plot_path)
    
    return plot_path

def plot_break_even_period(p, r, N, n, new_r, new_n, c, co):
    
    '''
    Plot the break-even period for mortgage refinancing.
    Defined as the number of months it takes for the cumulative monthly savings from refinancing to equal or exceed the closing costs.
    '''
    
    refinance_results = mortgage_refinance_calculation(p, r, N, n, new_r, new_n, c, co)
    
    new_remaining_term = new_n
    closing_cost = c
    monthly_savings = refinance_results['monthly_savings']
    
    cumulative_savings = [monthly_savings * i for i in range(1, (new_remaining_term  + 1))]
    
    # Break even point
    break_even_month = 0
    for i, savings in enumerate(cumulative_savings):
        if savings >= c:
            break_even_month = i + 1
            break
        
    plt.figure(figsize = (15, 6))
    plt.plot(range(1, new_remaining_term + 1), cumulative_savings, label = 'Cumulative Savings', color = 'blue')
    plt.axhline(y = closing_cost, color = 'r', linestyle = '--', label = 'Closing Cost')
    plt.axvline(x = break_even_month, color = 'g', linestyle = '--', label = 'Break-even Month')
    plt.scatter(break_even_month, closing_cost, color = 'black', zorder = 5, label = 'Break-even Point') # highlight break-even point
    plt.annotate(f'Month {break_even_month}', 
                (break_even_month, closing_cost), 
                textcoords = 'offset points', 
                xytext = (-15,15), ha = 'center', fontsize = 12,
                arrowprops = dict(arrowstyle = "->", lw = 1.5))
    plt.xlabel('Months')
    plt.ylabel('Cumulative Savings ($)')
    plt.legend()
    plt.grid(True)
    plt.title('Break-even Period for Mortgage Refinancing')
    
    plot_path = os.path.join('static', 'break_even_period.png')
    plt.savefig(plot_path)
    
    return plot_path

def plot_savings_interval(p, r, N, n, new_r, new_n, c, co, year_interval = 7):
    
    '''
    Plot the cumulative savings for mortgage refinancing over a specified interval of years.
    
    Note:
    - The closing cost is assumed to be paid in advance.
    '''
    
    refinance_results = mortgage_refinance_calculation(p, r, N, n, new_r, new_n, c, co)
    monthly_savings = refinance_results['monthly_savings']
    
    # Cumulative savings cashflow
    cashflow = [-c]
    cashflow.extend([monthly_savings] * new_n)
    cumulative_cashflow = np.cumsum(cashflow)
    
    # Display: 0 for closing cost, first year, intervals, last year
    indices_display = [0, 12] + list(range(12 * year_interval, new_n + 1, 12 * year_interval))
    indices_display = [i for i in indices_display if i <= new_n]
    if indices_display[-1] != new_n:
        indices_display.append(new_n)

    y = list(reversed(range(len(indices_display))))  # Reverse the y-values
    x = [cumulative_cashflow[i] for i in indices_display]
    labels = ['Cost'] + [f'Year {i // 12}' for i in indices_display[1:]]
    colors = ['red' if val < 0 else 'blue' for val in x]
    
    plt.figure(figsize = (15, 6))
    bars = plt.barh(y, x, tick_label = labels, color = colors)
    plt.ylabel('Year')
    plt.xlabel('Cumulative Savings ($)')
    formatter = FuncFormatter(thousands_formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    for bar, value in zip(bars, x):
        if value < 0:
            plt.text(max(x) * 0.08, bar.get_y() + bar.get_height()/2 , f'${value:,.0f}', va = 'center', ha = 'right', fontsize = 14)
        else:
            plt.text(value + max(x) * 0.01, bar.get_y() + bar.get_height()/2 , f'${value:,.0f}', va = 'center', ha = 'left', fontsize = 14)
    plt.title('Total Cumulative Savings for Mortgage Refinancing')
    plt.grid(True, axis = 'x')
    plt.tight_layout()
    
    plot_path = os.path.join('static', 'plot_savings_interval.png')
    plt.savefig(plot_path)
    
    return plot_path


#-------------------------Home-----------------------
@app.route('/', methods=['POST'])
def calculate_refinance():

        discount_rate = get_discount() / 100
        p     = float(request.form['p'])
        r     = float(request.form['r'])
        N     = int(request.form['N'])
        n     = int(request.form['n'])
        new_r = float(request.form['new_r'])
        new_n = int(request.form['new_n'])
        c     = float(request.form['c'])
        co    = float(request.form['co'])
        result =  mortgage_refinance_calculation(p, r, N, n, new_r, new_n, c, co)
        result1 =  mortgage_refinance_total_costs_calculation(p, r, N, n, new_r, new_n, c, co, discount_rate)
        df = loan_amortization_schedule(p, new_r, new_n)
        df = df.round(2)
        table_html = df.to_html(index=False, classes="amortization-table")

        plot1_path = plot_monthly_payment(p, r, N, n, new_r, new_n, c, co)
        plot2_path = plot_loan_amortization_schedule(p, r, N, n, new_r, new_n, c, co)
        plot3_path = plot_break_even_period(p, r, N, n, new_r, new_n, c, co)
        plot4_path = plot_savings_interval(p, r, N, n, new_r, new_n, c, co, year_interval = 7)
    

        rate_df =  get_mortgagerate()

        return render_template('mortgage_calculator.html', result=result, result1=result1, 
                               description=description, 
                               rate_data=rate_df, table_html=table_html, plot1_path=plot1_path, 
                               plot2_path=plot2_path, plot3_path=plot3_path, plot4_path=plot4_path)

@app.route("/")
def home():
    rate_df = get_mortgagerate()
    return render_template("mortgage_calculator.html", rate_data=rate_df, description=description)



@app.route('/mortgages')
def mortgages():
    return render_template('mortgages.html')


if __name__ == '__main__':
    app.run(debug=True)




