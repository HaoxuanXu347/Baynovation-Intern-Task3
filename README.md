# Baynovation-Intern-Task3

The main functionality of this application will involve allowing users to input relevant information such as current loan details, interest rates, loan term, closing costs, and any potential new loan details.

Once the user provides this information, the application will generate a graphical representation of the refinance break-even point. This graph will automatically adjust itself as users change their input values, providing them with a clear visualization of when it makes financial sense for them to refinance their loans.

Your main responsibilities will include:

1. Development: Design and develop the web application/GUI using the appropriate technologies and frameworks. Ensure that the application provides a smooth user experience and accurately calculates and displays the refinance break-even point.

2. User Interface Design: Create an intuitive and user-friendly interface that guides users through the input process and displays the graph in a clear and understandable manner.

3. Dynamic Plotting: Implement the feature that automatically adjusts the break-even point plot in real-time as users modify their input values.

4. Testing and Optimization: Thoroughly test the application to identify and fix any bugs or issues. Optimize the performance of the application to ensure quick responsiveness.

5. Documentation: Prepare documentation that outlines how the application works, its features, and any technical details that may be relevant for future maintenance.

rate_df = get_mortgagerate() 

<table style="position: absolute; right: 50px; top: 200px;">
        <thead>
            <tr>
                <th>Rate Type</th>
                <th>Rate</th>
            </tr>
        </thead>
        <tbody>
            {% for index, row in rate_data.iterrows() %}
            <tr>
                <td>{{ row['rate_type'] }}</td>
                <td>{{ row['rate'] }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

       <img src="/static/logo.jpeg" alt="Baynovation Logo" 
    style="position: absolute; left: 30px; top: 80px; font-size: small; border-radius: 100%;">