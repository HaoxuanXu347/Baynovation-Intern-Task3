# Baynovation-Intern-Task3

Credit: Shuyan Kang

The main functionality of this application will involve allowing users to input relevant information such as current loan details, interest rates, loan term, closing costs, and any potential new loan details.

Once the user provides this information, the application will generate a graphical representation of the refinance break-even point. This graph will automatically adjust itself as users change their input values, providing them with a clear visualization of when it makes financial sense for them to refinance their loans.

Your main responsibilities will include:

1. Development: Design and develop the web application/GUI using the appropriate technologies and frameworks. Ensure that the application provides a smooth user experience and accurately calculates and displays the refinance break-even point.

2. User Interface Design: Create an intuitive and user-friendly interface that guides users through the input process and displays the graph in a clear and understandable manner.

3. Dynamic Plotting: Implement the feature that automatically adjusts the break-even point plot in real-time as users modify their input values.

4. Testing and Optimization: Thoroughly test the application to identify and fix any bugs or issues. Optimize the performance of the application to ensure quick responsiveness.

5. Documentation: Prepare documentation that outlines how the application works, its features, and any technical details that may be relevant for future maintenance.



 rate_df =  get_mortgagerate()

        return render_template('mortgage_calculator.html', result=result, result1=result1, 
                               description=description, 
                               rate_data=rate_df, table_html=table_html, plot1_path=plot1_path, 
                               plot2_path=plot2_path, plot3_path=plot3_path, plot4_path=plot4_path)

@app.route("/")
def home():
    rate_df = get_mortgagerate()
    return render_template("mortgage_calculator.html", rate_data=rate_df, description=description)


      <div style="display: flex; justify-content: space-around; position: absolute; top: 500px; left: 35%; width: 60%;">
                    <table style="width: 100%; border-collapse: collapse; padding: 10px; border: 1px solid #dddddd; text-align: center; font-family: Arial, sans-serif;">
                        <thead>
                            <tr style="background-color: #f2f2f2;">
                                <th style="padding: 15px; border: 1px solid #dddddd; text-align: center;">Rate Type</th>
                                {% for index, row in rate_data.iterrows() %}
                                <th style="padding: 15px; border: 1px solid #dddddd; text-align: center;">{{ row['rate_type'] }}</th>
                                {% endfor %}
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td style="padding: 15px; border: 1px solid #dddddd; text-align: center; font-weight: bold;">Rate</td>
                                {% for index, row in rate_data.iterrows() %}
                                <td style="padding: 15px; border: 1px solid #dddddd; text-align: center;">{{ row['rate'] }}</td>
                                {% endfor %}
                            </tr>
                        </tbody>
                    </table>
                </div>




          

