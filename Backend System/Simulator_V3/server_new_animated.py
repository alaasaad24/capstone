from tkinter import *
from tkinter import ttk
import threading
import tkinter as tk
from tkinter import font

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import base64
from flask import Flask, send_file, request, jsonify
# from flask_cors import CORS

app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

#using thread to run GUI and Flask server 

# Tkinter GUI
root = Tk()
root.title("UTRADE Simulator")
# root.geometry("700x350") #500x800")  
# root.resizable(0, 0)

# Tkinter GUI component 
custom_font = font.Font(size=14,weight='bold')
TTitle = Label(root, text="UTRADE Simulator Form",font=custom_font)
TTitle.pack(padx=20, pady=10)                   #.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set window dimensions
window_width = 500
window_height = 800

# Calculate x and y positions for centering the window
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

#geometry to center it on the screen
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# configure the grid
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)

# # Calculate screen width and height
# user32 = ctypes.windll.user32
# screen_width = user32.GetSystemMetrics(0)
# screen_height = user32.GetSystemMetrics(1)

# # Set window dimensions
# window_width = 500
# window_height = 800

# # Calculate x and y positions for centering the window
# x_position = (screen_width - window_width) // 2
# y_position = (screen_height - window_height) // 2

# # Set window geometry to center it on the screen
# root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

def run_flask():
    app.run()
    
def run_animation():
    # Plotting the stock chart and simulation
    simulate_plot()

def get_value_GUI (even=NONE):
    global start_train_year,start_year, end_year,previous_days_for_trend, symbol
    symbol = stroke_combobox.get()
    start_train_year= startTrain.get()
    start_year=start.get()
    end_year= end.get()
    previous_days_for_trend= trend.get()
    print(f"Parameters: \nsymbol: {symbol}\nstart_train_year: {start_train_year}\nstart_test_year: {start_year}\nend_test_year: {end_year}\ntrend: {previous_days_for_trend}")
    # label = Label(root, text="Start train:" + start_train_year )
    # label.grid()
    # label = Label(root, text="Start testing:" + start_year )
    # label.grid()
    # label = Label(root, text="Symbol:" + symbol )
    # label.grid()
    process_data_and_routes()
    
def process_data_and_routes():
    # Load the data
    # symbol = "MSFT"
    data = pd.read_csv(f"Stock_Historical_Data/{symbol}.csv")
    data = data.set_index('Date', drop=True)
    
    # Check if the index is a DatetimeIndex, if not, convert it
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.set_index(pd.to_datetime(data.index, utc=True))
    
    # Filter the data based on user input
    global start_train_year, start_year, end_year, previous_days_for_trend
    start_train_year = int(start_train_year) if start_train_year else None
    start_year = int(start_year) if start_year else None
    end_year = int(end_year) if end_year else None
    previous_days_for_trend = int(previous_days_for_trend) if previous_days_for_trend else None
    
    train_data = []
    test_data = []
    
    # Process data based on user input
    if start_year is not None and end_year is not None:
        if start_year > 1900 and end_year > 1900:
            train_data = data[(data.index.year >= start_train_year) & (data.index.year < start_year)].copy()
            test_data = data[(data.index.year >= start_year) & (data.index.year < end_year)].copy()
            if previous_days_for_trend != 0:
                test_data = pd.concat([train_data.iloc[-previous_days_for_trend:], data[(data.index.year >= start_year) & (data.index.year < end_year)].copy()])
        else:
            train_data = data.iloc[0:-250]
            testing_days = 250 + previous_days_for_trend
            test_data = data.iloc[-testing_days:]
    
    # Return data
    return train_data, test_data


def get_test_data():
    _, test_data = process_data_and_routes()
    return test_data

# Define Flask routes1
@app.route('/get_data')
def get_data():
    try:
        train_data, test_data = process_data_and_routes()
        
        # Serialize the DataFrames to base64-encoded strings
        serialized_train_data = serialize_dataframe_to_base64(train_data)
        serialized_test_data = serialize_dataframe_to_base64(test_data)

        return {
            'train_data': serialized_train_data,
            'test_data': serialized_test_data
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500
    
# Define Flask routes2
@app.route('/receive_results', methods=['POST'])
def receive_results():
    try:
        # Receive the results DataFrame from the client
        results_data = request.form
        model_accuracy = results_data['model_accurecy']
        serialized_results = results_data['Prediction']

        # Decode base64 strings and unpickle DataFrame
        decoded_results = base64.b64decode(serialized_results)
        results_df = pickle.loads(decoded_results)
        

        # Update the accuracy label text with the received model accuracy
        accuracy_label.config(text=f"Model Accuracy:\n {model_accuracy}")
        
        # print("printing results_df:")
        results_df = pd.DataFrame(results_df)
        results_df['Date'] = pd.to_datetime(results_df['Date'])
        results_df.set_index('Date', inplace=True)
        # print(results_df.head())
        
        
        the_test_data = get_test_data()
        the_test_data = the_test_data[previous_days_for_trend:]
        # print("The_test_data")
        # print(the_test_data.head())
        if the_test_data.shape[0] > results_df.shape[0]:
            return "you are sending missing data, you may drop some data, try to increase 'previous_days_for_trend'"
        else:
            # Plotting the prediction graph
            simulate_plot(results_df, the_test_data)

        return "Results received successfully!"
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500
    

# Start Flask server in a separate thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()


def serialize_dataframe_to_base64(df):
    # Serialize the DataFrame 
    serialized_df = pickle.dumps(df)
    
    # Encode the serialized DataFrame as base64
    base64_encoded_df = base64.b64encode(serialized_df).decode('utf-8')
    
    return base64_encoded_df




def simulate_plot(results_df, test_data):
    
    def plot_in_main_thread():
        #Plotting the stock chart
        plt.figure(figsize=(8,4))
        plt.plot(test_data['Close'], label='Closing Price')

        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

        initial_cash = 100000  # Set your initial cash amount
        cash = initial_cash
        stocks_held = 0   # Number of stocks
        portfolio_value = []

        for index, row in test_data.iterrows():

            # Convert the index to its integer representation
            index_loc = results_df.index.get_loc(index)
            # Access the corresponding row in results_df
            prediction = int(results_df.iloc[index_loc].Predicted_Result)

            if prediction == 1:  # Buy signal
                while cash > row['Close']:
                    stocks_held += 1
                    cash -= row['Close']
            elif prediction == -1:  # Sell signal
                while stocks_held > 0:
                    cash += row['Close']
                    stocks_held -= 1

            portfolio_value.append(cash + stocks_held * row['Close'])

        # Include the last portfolio value after the last data point
        portfolio_value.append(cash + stocks_held * test_data.iloc[-1]['Close'])

        final_portfolio_value = portfolio_value[-1]  # Use the last value in the list
        gain_or_loss = final_portfolio_value - initial_cash

        Results = ""
        Results += f"Initial Portfolio Value: {"{:,}".format(round(initial_cash, 2))}$\n"
        Results += f"Final Portfolio Value: {"{:,}".format(round(final_portfolio_value, 2))}$\n"
        Results += f"Gain or Loss: {"{:,}".format(round(gain_or_loss, 2))}$"
        
        Prais_label.config (text=f"{Results}")

        # accuracy_label.config(text= f'{Results}\n\n')
        # print(Results)

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        line, = ax.plot([], [], label='Portfolio Value', color='blue')
        initial_cash_line = ax.axhline(y=initial_cash, color='r', linestyle='--', label='Initial Cash')
        ax.set_xlabel('Time')
        ax.set_ylabel('Portfolio Value')
        ax.set_title('Stock Trading Simulation')
        ax.legend()

        
        def init():
            ax.set_xlim(0, len(portfolio_value))
            ax.set_ylim(min(portfolio_value) * 0.9, max(portfolio_value) * 1.1)
            return line, initial_cash_line

        def update(frame):
            x_data = list(range(1, frame + 1))
            y_data = portfolio_value[:frame]
            line.set_data(x_data, y_data)
            if frame == len(portfolio_value):
                ani.event_source.stop()  # Stop animation when last point is reached
            return line, initial_cash_line

        ani = FuncAnimation(fig, update, frames=len(portfolio_value) + 1, init_func=init, blit=True, interval=35) #animation speed (interval)
        plt.show()


        # Reset the warning filter after the code block if needed
        warnings.resetwarnings()

    # Schedule the plotting function to run in the main thread
    root.after(0, plot_in_main_thread)

def clearFunction():
    stroke_combobox.delete(0, END)
    startTrain.delete(0, END)
    start.delete(0, END)
    end.delete(0, END)
    trend.delete(0, END)
    
# Tkinter GUI component
# Stock selection
stroke_frame = Frame(root)
stroke_frame.pack(padx=20, pady=10)

stroke_label = Label (stroke_frame,         text="       Select stock:            ", font=("Helvetica", 13)) 
stroke_label.pack(side=LEFT, padx=20, pady=10)


stroke_options = ['', 'AAPL', 'MSFT', 'TSLA', 'META', 'IBM', 'NVDA', 'AMZN', 'GOOGL', 'INTC', '^IXIC']
stroke_combobox = ttk.Combobox(stroke_frame, values=stroke_options,width=20)
stroke_combobox.pack(side=LEFT,padx=20, pady=10)
stroke_combobox.configure(font=("Helvetica", 12))

stroke_combobox.current(0)  # Default selection
stroke_combobox.bind("<<ComboboxSelected>>", get_value_GUI)

startTrain_frame = Frame(root)
startTrain_frame.pack(padx=20, pady=10)

startTrain_label = Label(startTrain_frame, text="Select Start Training Year:",font=("Helvetica", 13))
startTrain_label.pack(side=LEFT ,padx=20, pady=10)

startTrain = Entry(startTrain_frame,width=20) # ipady=3
startTrain.pack(side=LEFT,padx=20, pady=10)
startTrain.configure(font=("Helvetica", 12))

start_frame = Frame(root)
start_frame.pack(padx=20, pady=10)

start_label = Label(start_frame, text="Select Start Testing Year: ",font=("Helvetica", 13) )
start_label.pack(side=LEFT,padx=20, pady=10)

start = Entry(start_frame,  width=20) # ipady=3
start.pack(side=LEFT,padx=20, pady=10)
start.configure(font=("Helvetica", 12))

end_frame = Frame(root)
end_frame.pack(padx=20, pady=10)

end_label = Label(end_frame, text="Select End Testing Year:   ",font=("Helvetica", 13))
end_label.pack(side=LEFT,padx=20, pady=10)

end = Entry(end_frame,  width=20) # ipady=3
end.pack(side=LEFT,padx=20, pady=10)
end.configure(font=("Helvetica", 12))

trend_frame = Frame(root)
trend_frame.pack(padx=20, pady=10)

trend_label = Label(trend_frame, text="previous days data:        ",font=("Helvetica", 13))
trend_label.pack(side=LEFT,padx=20, pady=10)

trend= Entry(trend_frame,  width=20) # ipady=3
trend.pack(side=LEFT ,padx=20, pady=10)
trend.configure(font=("Helvetica", 12))

button_frame = Frame(root)
# button_frame.pack(anchor="w", padx=100, pady=30) #TMMM
button_frame.pack(anchor="w", padx=100, pady=(0, 10))

save=Button(button_frame, command=get_value_GUI, text="save parameters",bg="#77C3EF" ,width=13, height=2, font=("Helvetica", 13)) #font="30",
# save.pack(side=RIGHT, padx=(0, 40))
save.pack(side=LEFT, padx=(0, 5))
# save.configure(font=("Helvetica", 12))tmaam

clear=Button(button_frame, command=clearFunction, text="clear",bg="#77C3EF",width=13, height=2, font=("Helvetica", 13)) # fg="Black"
clear.pack(side=LEFT, padx=(100, 0)) # padx=(0,50))#,padx=(0, 20))tmaaam
#clear.configure(font=("Helvetica", 12))

#to display model accuracy
accuracy_label = Label(root, text="",width=50)
accuracy_label.pack(padx=20, pady=10)
accuracy_label.configure(font=("Helvetica", 12))

Prais_label = Label(root, text="",width=50)
Prais_label.pack(padx=20, pady=10)
Prais_label.configure(font=("Helvetica", 12))

# Run Tkinter GUI
root.mainloop()