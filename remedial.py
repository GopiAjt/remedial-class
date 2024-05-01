import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import ttk  # for progress bar
import smtplib
import time  # for delay
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Load the data from an Excel file
df = pd.read_excel('Remedial Dataset.xlsx')

# Separate numeric and non-numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns
non_numeric_columns = df.select_dtypes(exclude=['number']).columns

# Handle missing values for numeric columns by filling them with the mean of the respective columns
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

# Handle missing values for non-numeric columns by filling them with the most frequent value
non_numeric_imputer = SimpleImputer(strategy='most_frequent')
df[non_numeric_columns] = non_numeric_imputer.fit_transform(df[non_numeric_columns])

# Encode categorical variables
label_encoder = LabelEncoder()
df['Extra curricular'] = label_encoder.fit_transform(df['Extra curricular'])
df['Placements Status'] = label_encoder.fit_transform(df['Placements Status'])

# Define the features and target variable
features = ['1st Year INA1', '1st Year INA2', '2nd Year INA1', '2nd Year INA2', 
            'No. of Backlogs', 'Extra curricular', 'Placements Status']
target = 'Slow Learner'

# Assume a student is a slow learner if any of the INA scores is less than 20, they have backlogs,
# or they didn't participate in extracurricular activities or didn't get placed
df[target] = df.apply(lambda row: 1 if ((row['1st Year INA1'] < 20) or
                                         (row['1st Year INA2'] < 20) or
                                         (row['2nd Year INA1'] < 20) or
                                         (row['2nd Year INA2'] < 20) or 
                                         (row['No. of Backlogs'] > 0) or
                                         (row['Extra curricular'] == 0) or
                                         (row['Placements Status'] == 0)) else 0, axis=1)


# Create the Decision Tree model
dt_model = DecisionTreeClassifier()

# Train the Decision Tree model on the whole dataset
dt_model.fit(df[features], df[target])

# Create the Naive Bayes model
nb_model = GaussianNB()

# Train the Naive Bayes model on the whole dataset
nb_model.fit(df[features], df[target])

# Load test data from another Excel file
test_df = pd.read_excel('test-remedial.xlsx')

# Handle missing values and encode categorical variables for test data similar to the training data (above)

# Add a 'Slow Learner' prediction column to the test data using the Decision Tree model
test_df['dt_Slow Learner'] = dt_model.predict(test_df[features])

# Add a 'Slow Learner' prediction column to the test data using the Naive Bayes model
test_df['nb_Slow Learner'] = nb_model.predict(test_df[features])

def display_data(data):
  """Displays data in a GUI window with labels and frames, with borders around columns"""
  window = tk.Tk()
  window.title("Slow Learner Prediction Results")

  # Create a Frame for the table
  table_frame = tk.Frame(window)
  table_frame.pack(padx=10, pady=10)

  # Configure border style for columns
  border_width = 1
  relief_style = 'groove'  # Choose border style ('flat', 'groove', 'raised', 'ridge', 'sunken', 'inset')

  # Create headers for columns (including index)
  headers = list(data.columns)
  for col_index, col_name in enumerate(headers):
    header_label = tk.Label(table_frame, text=col_name, font=("Arial", 10, "bold"))
    header_label.grid(row=0, column=col_index, sticky="nsew", padx=border_width, pady=border_width)
    header_label.config(relief=relief_style, highlightthickness=border_width, highlightbackground="black")

  # Enumerate through rows (including index)
  for row_index, row in data.iterrows():
    for col_index, value in enumerate(row):
      cell_label = tk.Label(table_frame, text=str(value), font=("Arial", 8))
      cell_label.grid(row=row_index+1, column=col_index, sticky="nsew", padx=border_width, pady=border_width)
      cell_label.config(relief=relief_style, highlightthickness=border_width, highlightbackground="black")


    # Create a Frame for buttons (optional for better organization)
  button_frame = tk.Frame(window)
  button_frame.pack(padx=10, pady=10)

  def button1_clicked():
    # Filter slow learners based on the 'Slow Learner' column
    slow_learners = data[data['Slow Learner'] == 1]['Email ID']

    # Create a new window for the loading screen
    loading_window = tk.Toplevel(window)
    loading_window.title("Sending Emails...")

    # Create a progress bar
    progress = ttk.Progressbar(loading_window, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
    progress.pack(padx=10, pady=10)
    progress.start(interval=100)  # Update progress bar periodically

    time.sleep(5)

    # Send email to each slow learner (replace with your email configuration)
    sender_email = "capturenow.in@gmail.com"  # Replace with your email address
    sender_password = "ggasmkitfqdibsal"  # Replace with your email password,
    smtp_server = "smtp.gmail.com"  # Replace with your SMTP server (e.g., 'smtp.gmail.com')
    port = 587  # Replace with your SMTP port (e.g., 587 for Gmail)

    message = """
    Dear [Student Name] Parents/guardians,

    This email is to inform you that you have been identified as a potential slow learner based on your academic performance. 
    We encourage you to seek academic support services available at the institution to improve your learning outcomes.

    Sincerely,
    [Your Name/Institution Name]
    """

    with smtplib.SMTP(smtp_server, port) as server:
      server.starttls()
      server.login(sender_email, sender_password)
      for email_id in slow_learners:
        progress.step(10)  # Update progress bar for each email sent
        receiver_email = email_id
        server.sendmail(sender_email, receiver_email, message.replace("[Student Name]", data[data['Email ID'] == email_id]['Name'].values[0]))

    progress.stop()  # Stop progress bar after all emails sent
    confirmation_message = tk.Label(loading_window, text="Emails sent successfully to slow learners.")
    confirmation_message.pack(padx=10, pady=10)

    # Close the loading window after a short delay
    loading_window.after(2000, loading_window.destroy)  # Close after 2 seconds


  def button2_clicked():
    # Filter slow learners based on the 'Slow Learner' column
    slow_learners = data[data['Slow Learner'] == 1]['Email ID']

    # Create a new window for the loading screen
    loading_window = tk.Toplevel(window)
    loading_window.title("Sending Emails...")

    # Create a progress bar
    progress = ttk.Progressbar(loading_window, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
    progress.pack(padx=10, pady=10)
    progress.start(interval=100)  # Update progress bar periodically

    time.sleep(5)

    # Send email to each slow learner (replace with your email configuration)
    sender_email = "capturenow.in@gmail.com"  # Replace with your email address
    sender_password = "ggasmkitfqdibsal"  # Replace with your email password,
    smtp_server = "smtp.gmail.com"  # Replace with your SMTP server (e.g., 'smtp.gmail.com')
    port = 587  # Replace with your SMTP port (e.g., 587 for Gmail)

    message = """
    Dear [Student Name],

    This email is to inform you that you have been identified as a Non placed candidate. 
    We encourage you to seek start applying for off campus job applications.

    You can find the application links billow:

    {https://www.accenture.com/us-en/careers}
    {https://www.ibm.com/careers}
    {https://careers.cognizant.com/global/en}
    {https://www.infosys.com/careers/apply.html}
    {https://careers.wipro.com/}
    {https://www.google.com/about/careers/applications/}
    {https://careers.microsoft.com/}

    Sincerely,
    [Your Name/Institution Name]
    """

    with smtplib.SMTP(smtp_server, port) as server:
      server.starttls()
      server.login(sender_email, sender_password)
      for email_id in slow_learners:
        progress.step(10)  # Update progress bar for each email sent
        receiver_email = email_id
        server.sendmail(sender_email, receiver_email, message.replace("[Student Name]", data[data['Email ID'] == email_id]['Name'].values[0]))

    progress.stop()  # Stop progress bar after all emails sent
    confirmation_message = tk.Label(loading_window, text="Emails sent successfully to slow learners.")
    confirmation_message.pack(padx=10, pady=10)

    # Close the loading window after a short delay
    loading_window.after(2000, loading_window.destroy)  # Close after 2 seconds

  # Create buttons
  button1 = tk.Button(button_frame, text="send mail to Parents", command=button1_clicked)
  button2 = tk.Button(button_frame, text="sent placement links", command=button2_clicked)

  # Position buttons within the button frame (optional for specific layout)
  button1.grid(row=0, column=0, padx=5, pady=5)
  button2.grid(row=0, column=1, padx=5, pady=5)

      # Create the plot
  x = [1, 2, 3]
  y = [2, 4, 1]
  fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
  ax.plot(x, y)
  ax.set_xlabel('x - axis')
  ax.set_ylabel('y - axis')
  ax.set_title('My first graph!')

    # Create a canvas to display the graph in the window
  canvas_graph = FigureCanvasTkAgg(fig, master=window)
  canvas_widget = canvas_graph.get_tk_widget()

    # Pack the graph below the buttons
  canvas_widget.pack(padx=10, pady=10)
  window.mainloop()


# Create 'Slow Learner' column based on model predictions
test_df['Slow Learner'] = (test_df['dt_Slow Learner'] == 1) | (test_df['nb_Slow Learner'] == 1)
test_df['Remidial Classes Needed'] = (test_df['dt_Slow Learner'] == 1) | (test_df['nb_Slow Learner'] == 1)
print(test_df)


# Assuming you have your test data in 'test_df' after preprocessing
display_data(test_df)
