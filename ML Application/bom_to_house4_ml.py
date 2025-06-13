# TensorFlow ML Application - BOM Weather to House 4 Energy Prediction
# Adapted from Google Colab code for local Windows environment
# Uses temperature from BOM data to predict energy consumption in House 4

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import math
import pickle
import matplotlib.pyplot as plt

# Set up local data paths (replacing Google Colab paths)
script_dir = os.path.dirname(os.path.abspath(__file__))
datapath = os.path.join(script_dir, 'Datasets')

bomdata = os.path.join(datapath, 'BOM_year.csv')
house4data = os.path.join(datapath, 'House 4_Melb West.csv')

# Create processed data files
bomfile = os.path.join(script_dir, 'bom.dat')
house4file = os.path.join(script_dir, 'house4.dat')

#function to change month from string to number
def mTon(m):
    m = m.lower()
    if m == 'jan':
        return '01'
    elif m == 'feb':
        return '02'
    elif m == 'mar':
        return '03'
    elif m == 'apr':
        return '04'
    elif m == 'may':
        return '05'
    elif m == 'jun':
        return '06'
    elif m == 'jul':
        return '07'
    elif m == 'aug':
        return '08'
    elif m == 'sep':
        return '09'
    elif m == 'oct':
        return '10'
    elif m == 'nov':
        return '11'
    elif m == 'dec':
        return '12'
    else:
        return '00'

def process_bom_data():
    """Process BOM weather data"""
    print("Processing BOM data...")
    
    # read and pre-process bom data
    df = pd.read_csv(bomdata)
    bomarr = df.to_numpy()
    bom = {}
    dates = []
    
    for line in bomarr:
        d, m, y = line[0].split('-')
        m = mTon(m)
        key = y + m + d
        dates.append(int(key))
        data = []
        flag = True
        
        # Load columns: MinTemp (index 1), MaxTemp (index 2), 9am temp (index 4), 3pm temp (index 5)
        temp_indices = [1, 2, 4, 5]  # MinTemp, MaxTemp, 9amTemp, 3pmTemp
        for i in temp_indices:
            if i < len(line):
                tmp = float(line[i]) if not pd.isna(line[i]) else float('nan')
                if math.isnan(tmp):  # any NaN will drop the record
                    flag = False
                data.append(tmp)
            else:
                flag = False
        
        if flag and len(data) == 4:  # Ensure we have all 4 temperature values
            bom[key] = data
    
    print(f"BOM data processed: {len(bom)} valid records")
    
    #save the processed data to dat file
    with open(bomfile, 'wb') as file:
        pickle.dump(bom, file)
    
    return bom

def process_house4_data():
    """Process House 4 energy data"""
    print("Processing House 4 data...")
    
    #read and pre-process house 4 data
    df = pd.read_csv(house4data, header=None, names=['DateTime', 'Energy'])
    house4arr = df.to_numpy()
    house4 = {}
    house4data_processed = {}
    
    for item in house4arr:
        try:
            date_time = str(item[0])
            date, time = date_time.split(' ')
            d, m, y = date.split('-')
            
            # Convert 2-digit year to 4-digit year
            if len(y) == 2:
                y = '20' + y
            
            # Ensure 2-digit format for day and month
            d = d.zfill(2)
            m = m.zfill(2)
            
            h = int(time.split(':')[0])
            dates = y + m + d
            power = float(item[1])
            
            if math.isnan(power):
                continue  # skip the below, continue to next loop
            
            # Group power data for 9am (8-10am) and 3pm (2-4pm) periods
            if h >= 8 and h < 10:
                key = dates + '09'
                if key in house4.keys():
                    house4[key][0] = house4[key][0] + power
                    house4[key][1] = house4[key][1] + 1
                else:
                    house4[key] = [power, 1]
            elif h >= 14 and h < 16:
                key = dates + '15'
                if key in house4.keys():
                    house4[key][0] = house4[key][0] + power
                    house4[key][1] = house4[key][1] + 1
                else:
                    house4[key] = [power, 1]
        except Exception as e:
            continue  # Skip problematic rows
    
    # Calculate average power for each time period
    for key in house4.keys():
        house4data_processed[key] = house4[key][0] / house4[key][1]  # save the average power
    
    print(f"House 4 data processed: {len(house4data_processed)} valid records")
    
    #save house 4 processed data
    with open(house4file, 'wb') as file:
        pickle.dump(house4data_processed, file)
    
    return house4data_processed

def load_processed_data():
    """Load processed data from dat files"""
    print("Loading processed data...")
    
    # Check if processed files exist, if not create them
    if not os.path.exists(bomfile):
        bom = process_bom_data()
    else:
        with open(bomfile, 'rb') as file:
            bom = pickle.load(file)
    
    if not os.path.exists(house4file):
        house4data_processed = process_house4_data()
    else:
        with open(house4file, 'rb') as file:
            house4data_processed = pickle.load(file)
    
    return bom, house4data_processed

def generate_training_data(bom, house4data_processed):
    """Generate full dataset for ML - FOUR factor learning: temperature features - power"""
    print("Generating training data...")
    
    x_train_full = []
    y_train_full = []    # Four factor: MinTemp, MaxTemp, 9amTemp, 3pmTemp
    for k in bom.keys():
        k1 = k + '09'  # 9am data
        k2 = k + '15'  # 3pm data
        keys = house4data_processed.keys()
        
        if k1 in keys:
            # Use all 4 temperature features for 9am energy prediction
            if len(bom[k]) >= 4:
                x_train_full.append([bom[k][0], bom[k][1], bom[k][2], bom[k][3]])  # MinTemp, MaxTemp, 9amTemp, 3pmTemp
                y_train_full.append(house4data_processed[k1])
            else:
                print(f"Warning: BOM data for {k} has only {len(bom[k])} values, expected 4")
        
        if k2 in keys:
            # Use all 4 temperature features for 3pm energy prediction
            if len(bom[k]) >= 4:
                x_train_full.append([bom[k][0], bom[k][1], bom[k][2], bom[k][3]])  # MinTemp, MaxTemp, 9amTemp, 3pmTemp
                y_train_full.append(house4data_processed[k2])
            else:
                print(f"Warning: BOM data for {k} has only {len(bom[k])} values, expected 4")
    
    print(f"Generated {len(x_train_full)} training samples")
    return x_train_full, y_train_full

def create_train_test_split(x_train_full, y_train_full):
    """Separate test set and training set"""
    print("Creating train/test split...")
    
    #separate test set and training set
    part = int(len(x_train_full) * 0.05)
    x_test = x_train_full[-part:]
    y_test = y_train_full[-part:]
    
    x_train = x_train_full[:-part]
    y_train = y_train_full[:-part]
    
    x_test = np.array(x_test)
    y_test = np.array(y_test) / 1000.00   #divide by 1000 to show power in kw
    x_train = np.array(x_train)
    y_train = np.array(y_train) / 1000.00
      #reshape dataset to 3 dimensions for RNN
    x_train = x_train.reshape(len(x_train), 1, 4)  # 4 temperature features
    x_test = x_test.reshape(len(x_test), 1, 4)
    
    print(f"Training set: {x_train.shape}")
    print(f"Test set: {x_test.shape}")
    
    return x_train, x_test, y_train, y_test

def create_rnn_model():
    """Create RNN model"""
    print("Creating RNN model...")
    
    model = tf.keras.models.Sequential()
    # create 2 layers, 20 units per layer.
    # Each layer contains a state. This state is updated at each time step based on the previous state and the current input.
    # Each layer returns the entire sequence of outputs for next layer
    model.add(tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=(1, 4)))  # 4 temperature features
    model.add(tf.keras.layers.SimpleRNN(20, return_sequences=True))
    model.add(tf.keras.layers.Dense(1))
    
    #compile model and train the model
    loss = tf.keras.losses.MeanSquaredError() # use MSE as the loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # use Adam optimiser, set up learning rate
    
    model.compile(loss=loss, optimizer=optimizer) # compile the model configuration
    
    print("Model created and compiled!")
    return model

def train_model(model, x_train, y_train):
    """Train the model"""
    print("Training model...")
    
    batch_size = 1   # number of samples per iteration
    epochs = 30      # total number of iterations
    
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs) # train the model based on the selected parameters 
    
    return history

def evaluate_and_predict(model, x_test, y_test):
    """Evaluate model and make predictions"""
    print("Evaluating model...")
    
    #evaluate the model - result will be the mean square error
    model_result = model.evaluate(x_test, y_test)
    print(f"Model MSE: {model_result}")
    
    #predict test set
    predicted_power = model.predict(x_test)
    
    return predicted_power

def calculate_errors(predicted_power, y_test):
    """Calculate prediction errors"""
    print("Calculating prediction errors...")
    
    #calculate prediction error
    error = []
    error1k = []
    errorrate = []
    predicted_power_list = []
    
    for i in range(len(predicted_power)):
        predicted_power_list.append(predicted_power[i][0][0])
        e = predicted_power[i][0][0] - y_test[i]  # Fixed: use y_test[i] instead of y_test[0]
        error.append(e)
        error1k.append(1000 * e)
        if y_test[i] != 0:
            errorrate.append(100 * e / y_test[i])
        else:
            errorrate.append(0)
    
    print(f"Average error rate: {np.mean(errorrate):.2f}%")
    print(f"Mean absolute error: {np.mean(np.abs(error)):.4f} kW")
    
    return predicted_power_list, error, errorrate

def plot_results(y_test, predicted_power):
    """Graph the results"""
    print("Creating visualization...")
    
    # graph the results
    plt.figure(figsize=(12, 8))
    
    x = list(range(1, len(y_test) + 1))
    plt.plot(x, y_test, label='Actual kW', linewidth=2)
    plt.plot(x, [predicted_power[i][0][0] for i in range(len(predicted_power))], label='Predicted kW', linewidth=2)
    plt.legend(("Actual kW", "Predicted kW"), loc="upper right")
    plt.xlabel("# of data")
    plt.ylabel("Power (kW)")
    plt.title("Power Prediction - BOM Weather to House 4 Energy")
    plt.grid(True, alpha=0.3)
    plt.show()

def save_results(y_test, predicted_power_list, error, errorrate):
    """Save results to CSV"""
    results_df = pd.DataFrame({
        'Actual_kW': y_test,
        'Predicted_kW': predicted_power_list,
        'Error_kW': error,
        'Error_Rate_%': errorrate
    })
    
    results_file = os.path.join(script_dir, 'bom_to_house4_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")

def main():
    """Main function to run the complete ML application"""
    print("=== TensorFlow ML Application: BOM Weather → House 4 Energy Prediction ===")
    print("Using 4 temperature features (Min, Max, 9am, 3pm) to predict energy consumption\n")
    
    try:
        # Step 1: Load and process data
        bom, house4data_processed = load_processed_data()
        
        if len(bom) == 0 or len(house4data_processed) == 0:
            print("Error: No data could be processed!")
            return
        
        # Step 2: Generate training data
        x_train_full, y_train_full = generate_training_data(bom, house4data_processed)
        
        if len(x_train_full) == 0:
            print("Error: No overlapping data found!")
            return
        
        # Step 3: Create train/test split
        x_train, x_test, y_train, y_test = create_train_test_split(x_train_full, y_train_full)
        
        # Step 4: Create RNN model
        model = create_rnn_model()
        
        # Step 5: Train the model
        history = train_model(model, x_train, y_train)
        
        # Step 6: Evaluate and predict
        predicted_power = evaluate_and_predict(model, x_test, y_test)
        
        # Step 7: Calculate errors
        predicted_power_list, error, errorrate = calculate_errors(predicted_power, y_test)
        
        # Step 8: Create visualization
        plot_results(y_test, predicted_power)
        
        # Step 9: Save results
        save_results(y_test, predicted_power_list, error, errorrate)
        
        print("\n=== ML Application Completed Successfully ===")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
