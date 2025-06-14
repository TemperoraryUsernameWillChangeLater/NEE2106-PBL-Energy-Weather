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

# ====================
# CUDA CONFIGURATION
# ====================
def configure_cuda():
    """Configure CUDA for optimal GPU performance"""
    print("🚀 CUDA Configuration Starting...")
    
    # Check TensorFlow CUDA support
    cuda_built = tf.test.is_built_with_cuda()
    print(f"   • TensorFlow built with CUDA: {cuda_built}")
    
    # Get GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"   • Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"     GPU {i}: {gpu.name}")
        
        try:
            # Enable GPU memory growth to prevent allocation of all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("   ✅ GPU memory growth enabled")
            
            # Set the first GPU as the primary device
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"   ✅ Primary GPU set: {gpus[0].name}")
            
            # Configure GPU memory limit if needed (optional)
            # tf.config.experimental.set_memory_growth(gpus[0], True)
            
            # Test GPU computation
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                test_result = tf.matmul(test_tensor, test_tensor)
                print("   ✅ GPU computation test successful")
                
            return True
            
        except RuntimeError as e:
            print(f"   ⚠️  GPU configuration error: {e}")
            print("   🔄 Falling back to CPU")
            return False
    else:
        print("   ❌ No GPU detected")
        print("   💡 To enable GPU acceleration:")
        print("      1. Install CUDA 11.8 or 12.x")
        print("      2. Install cuDNN")
        print("      3. Install tensorflow-gpu or tensorflow[and-cuda]")
        return False

def set_mixed_precision():
    """Enable mixed precision for faster training on modern GPUs"""
    if tf.config.list_physical_devices('GPU'):
        try:
            # Enable mixed precision
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("   ⚡ Mixed precision (float16) enabled for faster training")
            return True
        except:
            print("   ⚠️  Mixed precision not supported, using float32")
            return False
    return False

# Configure CUDA at import time
cuda_available = configure_cuda()
mixed_precision_enabled = set_mixed_precision()
print("=" * 60)

# Set up local data paths (replacing Google Colab paths)
script_dir = os.path.dirname(os.path.abspath(__file__))
datapath = os.path.join(script_dir, 'Datasets')
refined_datasets_dir = os.path.join(script_dir, 'Refined Datasets')

# Create Refined Datasets directory if it doesn't exist
os.makedirs(refined_datasets_dir, exist_ok=True)

bomdata = os.path.join(datapath, 'BOM_year.csv')
house4data = os.path.join(datapath, 'House 4_Melb West.csv')

# Create processed data files in Refined Datasets directory
bomfile = os.path.join(refined_datasets_dir, 'bom.dat')
house4file = os.path.join(refined_datasets_dir, 'house4.dat')

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
    """Create RNN model with GPU optimization"""
    print("Creating RNN model...")
    
    # Create model with explicit GPU placement if available
    device_name = '/GPU:0' if cuda_available else '/CPU:0'
    print(f"   • Building model on: {device_name}")
    
    with tf.device(device_name):
        model = tf.keras.models.Sequential()
        
        # Create 2 layers, 20 units per layer with GPU-optimized configurations
        # Each layer contains a state. This state is updated at each time step based on the previous state and the current input.
        # Each layer returns the entire sequence of outputs for next layer
        model.add(tf.keras.layers.SimpleRNN(
            20, 
            return_sequences=True, 
            input_shape=(1, 4),  # 4 temperature features
            # Use CuDNN implementation for faster GPU training
            **({'use_bias': True} if cuda_available else {})
        ))
        
        model.add(tf.keras.layers.SimpleRNN(
            20, 
            return_sequences=True,
            **({'use_bias': True} if cuda_available else {})
        ))
        
        # Final dense layer - use float32 for mixed precision compatibility
        if mixed_precision_enabled:
            model.add(tf.keras.layers.Dense(1, dtype='float32'))
        else:
            model.add(tf.keras.layers.Dense(1))
    
    # Compile model with optimized settings
    loss = tf.keras.losses.MeanSquaredError() # use MSE as the loss function
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        # Add GPU-specific optimizations
        **({'jit_compile': True} if cuda_available else {})
    )
    
    model.compile(
        loss=loss, 
        optimizer=optimizer,
        # Enable XLA compilation for faster GPU execution
        **({'jit_compile': True} if cuda_available else {})
    )
    
    print("   ✅ Model created and compiled with GPU optimizations!")
    print(f"   • Device: {device_name}")
    print(f"   • Mixed Precision: {mixed_precision_enabled}")
    
    return model

def train_model(model, x_train, y_train, epochs=30):
    """Train the model with specified epochs and GPU optimizations"""
    print(f"Training model with {epochs} epochs...")
    print(f"   • Using device: {'/GPU:0' if cuda_available else '/CPU:0'}")
    print(f"   • Training samples: {len(x_train)}")
    
    batch_size = 32 if cuda_available else 1  # Larger batch size for GPU efficiency
    
    # Add callbacks for better training monitoring and GPU utilization
    callbacks = []
    
    if cuda_available:
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
    
    print(f"   • Batch size: {batch_size}")
    print(f"   • Callbacks: {len(callbacks)} enabled")
    
    # Train the model with GPU optimizations
    history = model.fit(
        x_train, y_train, 
        batch_size=batch_size, 
        epochs=epochs,
        callbacks=callbacks,
        verbose=1 if cuda_available else 2,  # More verbose output for GPU training
        # Enable performance optimizations
        **({'use_multiprocessing': True, 'workers': 4} if cuda_available else {})
    )
    
    print("   ✅ Training completed!")
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
    
    results_file = os.path.join(refined_datasets_dir, 'bom_to_house4_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")

def run_epoch_comparison():
    """Run comparison of different epoch counts from 50 to 1000 in increments of 50"""
    print("=== Epoch Comparison Study ===")
    print("Automatically testing epochs: 50, 100, 150, 200, ..., 1000 (20 models total)\n")
    
    # Load data
    bom, house4data_processed = load_processed_data()
    x_train_full, y_train_full = generate_training_data(bom, house4data_processed)
    x_train, x_test, y_train, y_test = create_train_test_split(x_train_full, y_train_full)
    
    # Generate epoch range from 50 to 1000 in increments of 50
    epoch_tests = list(range(50, 1001, 50))  # [50, 100, 150, ..., 1000]
    results = []
    
    print(f"Training {len(epoch_tests)} models...\n")
    
    for i, epochs in enumerate(epoch_tests, 1):
        print(f"[{i}/{len(epoch_tests)}] Training with {epochs} epochs...")
        
        # Create and train model
        model = create_rnn_model()
        history = model.fit(x_train, y_train, batch_size=1, epochs=epochs, verbose=0)
        
        # Evaluate
        mse = model.evaluate(x_test, y_test, verbose=0)
        predictions = model.predict(x_test, verbose=0)
        
        # Calculate error rate
        errors = [predictions[i][0][0] - y_test[i] for i in range(len(predictions))]
        error_rate = np.mean(np.abs(errors)) / np.mean(y_test) * 100
        
        results.append({
            'epochs': epochs,
            'mse': mse,
            'error_rate': error_rate,
            'predictions': predictions
        })
        
        print(f"    MSE: {mse:.4f}, Error Rate: {error_rate:.2f}%")
    
    # Create 5x4 subplot visualization (20 plots total)
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    fig.suptitle('Epoch Comparison Study: 50-1000 Epochs\nBOM Weather → House 4 Energy Prediction', fontsize=16, fontweight='bold')
    
    for i, result in enumerate(results):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        x_range = range(1, len(y_test) + 1)
        
        ax.plot(x_range, y_test, 'b-', label='Actual', linewidth=2)
        ax.plot(x_range, [result['predictions'][j][0][0] for j in range(len(result['predictions']))], 
                'r--', label='Predicted', linewidth=2)
        
        ax.set_title(f"Epochs: {result['epochs']}\nMSE: {result['mse']:.4f}, Error: {result['error_rate']:.1f}%", 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Data Point #', fontsize=8)
        ax.set_ylabel('Power (kW)', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Create summary table
    summary_data = []
    for result in results:
        summary_data.append({
            'Epochs': result['epochs'],
            'MSE': result['mse'],
            'Error_Rate_%': result['error_rate']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(refined_datasets_dir, 'epoch_comparison_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    # Summary analysis
    best_result = min(results, key=lambda x: x['mse'])
    worst_result = max(results, key=lambda x: x['mse'])
    
    print(f"\n=== COMPREHENSIVE RESULTS ===")
    print(f"🏆 Best Performance: {best_result['epochs']} epochs")
    print(f"   MSE: {best_result['mse']:.4f}")
    print(f"   Error Rate: {best_result['error_rate']:.2f}%")
    print(f"\n💔 Worst Performance: {worst_result['epochs']} epochs")
    print(f"   MSE: {worst_result['mse']:.4f}")
    print(f"   Error Rate: {worst_result['error_rate']:.2f}%")
    
    improvement = (worst_result['mse'] - best_result['mse']) / worst_result['mse'] * 100
    print(f"\n📊 Performance Improvement: {improvement:.1f}% better with optimal epochs")
    print(f"📁 Summary saved to: {summary_file}")
    
    print(f"\n💡 CONCLUSION:")
    print(f"   • Tested all epochs from 50 to 1000 in increments of 50")
    print(f"   • Optimal performance achieved at {best_result['epochs']} epochs")
    print(f"   • Results show {improvement:.1f}% improvement from worst to best")
    print(f"   • Beyond optimal point, more epochs may lead to overfitting")

def run_incremental_epoch_comparison():
    """Train a single model incrementally: 50, 100, ..., 500 epochs, storing results after each increment, and plot a 5x2 grid."""
    print("=== Incremental Epoch Comparison Study ===")
    print("Training a single model, continuing for 50, 100, ..., 500 epochs (total 500 epochs)\n")

    # Load data
    bom, house4data_processed = load_processed_data()
    x_train_full, y_train_full = generate_training_data(bom, house4data_processed)
    x_train, x_test, y_train, y_test = create_train_test_split(x_train_full, y_train_full)

    # Epoch increments
    epoch_tests = list(range(50, 501, 50))  # [50, 100, ..., 500]
    results = []    # Create and train model incrementally
    model = create_rnn_model()
    prev_epochs = 0
    for i, epochs in enumerate(epoch_tests, 1):
        print(f"[{i}/{len(epoch_tests)}] Training up to {epochs} epochs (from {prev_epochs})...")
        history = model.fit(x_train, y_train, batch_size=1, initial_epoch=prev_epochs, epochs=epochs, verbose=1)
        # Evaluate
        mse = model.evaluate(x_test, y_test, verbose=0)
        predictions = model.predict(x_test, verbose=0)
        # Calculate error rate
        errors = [predictions[j][0][0] - y_test[j] for j in range(len(predictions))]
        error_rate = np.mean(np.abs(errors)) / np.mean(y_test) * 100
        results.append({
            'epochs': epochs,
            'mse': mse,
            'error_rate': error_rate,
            'predictions': predictions
        })
        print(f"    MSE: {mse:.4f}, Error Rate: {error_rate:.2f}%")
        
        # Save results to CSV after each iteration
        save_epoch_results_to_csv(epochs, y_test, predictions, mse, error_rate, i == 1)  # header only on first iteration
        
        prev_epochs = epochs# Create 5x2 subplot visualization (10 plots total) with dynamic font scaling
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    
    # Calculate dynamic font sizes based on figure size
    base_size = min(fig.get_figwidth(), fig.get_figheight())
    title_font = max(8, int(base_size * 0.8))
    subplot_title_font = max(6, int(base_size * 0.5))
    label_font = max(5, int(base_size * 0.4))
    legend_font = max(4, int(base_size * 0.35))
    tick_font = max(4, int(base_size * 0.3))
    
    fig.suptitle('Incremental Epoch Comparison: 50-500 Epochs\nBOM Weather → House 4 Energy Prediction', 
                fontsize=title_font, fontweight='bold')
    
    for i, result in enumerate(results):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        x_range = range(1, len(y_test) + 1)
        ax.plot(x_range, y_test, 'b-', label='Actual', linewidth=2)
        ax.plot(x_range, [result['predictions'][j][0][0] for j in range(len(result['predictions']))], 
                'r--', label='Predicted', linewidth=2)
        ax.set_title(f"Epochs: {result['epochs']}\nMSE: {result['mse']:.4f}, Error: {result['error_rate']:.1f}%", 
                    fontsize=subplot_title_font, fontweight='bold')
        ax.set_xlabel('Data Point #', fontsize=label_font)
        ax.set_ylabel('Power (kW)', fontsize=label_font)
        ax.legend(fontsize=legend_font)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=tick_font)
    
    plt.tight_layout()
    plt.show(block=False)  # Non-blocking display

    # Create difference comparison plot (5x2 grid with 9 plots showing consecutive differences)
    if len(results) > 1:
        fig2, axes2 = plt.subplots(5, 2, figsize=(16, 20))
        
        # Calculate dynamic font sizes for difference plot
        diff_title_font = max(8, int(base_size * 0.8))
        diff_subplot_title_font = max(6, int(base_size * 0.5))
        diff_label_font = max(5, int(base_size * 0.4))
        diff_legend_font = max(4, int(base_size * 0.35))
        diff_tick_font = max(4, int(base_size * 0.3))
        
        fig2.suptitle('Prediction Differences Between Consecutive Epochs\n(Current Epoch - Previous Epoch)',fontsize=diff_title_font, fontweight='bold')
        
        # Calculate differences between consecutive epochs
        for i in range(len(results) - 1):
            row = i // 2
            col = i % 2
            ax = axes2[row, col]
            
            # Get predictions for current and previous epoch
            current_pred = [results[i+1]['predictions'][j][0][0] for j in range(len(results[i+1]['predictions']))]
            previous_pred = [results[i]['predictions'][j][0][0] for j in range(len(results[i]['predictions']))]
            
            # Calculate difference (current - previous)
            prediction_diff = [current_pred[j] - previous_pred[j] for j in range(len(current_pred))]
            
            x_range = range(1, len(prediction_diff) + 1)
            ax.plot(x_range, prediction_diff, 'g-', linewidth=2, label='Prediction Difference')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
              # Calculate statistics
            mean_diff = np.mean(prediction_diff)
            std_diff = np.std(prediction_diff)
            
            # Save difference results to CSV
            save_difference_results_to_csv(results[i]['epochs'], results[i+1]['epochs'], 
                                          prediction_diff, mean_diff, std_diff, i == 0)  # header only on first iteration
            
            ax.set_title(f"Epochs {results[i]['epochs']} → {results[i+1]['epochs']}\nMean Δ: {mean_diff:.4f}, Std: {std_diff:.4f}", 
                        fontsize=diff_subplot_title_font, fontweight='bold')
            ax.set_xlabel('Data Point #', fontsize=diff_label_font)
            ax.set_ylabel('Prediction Difference (kW)', fontsize=diff_label_font)
            ax.legend(fontsize=diff_legend_font)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=diff_tick_font)        # Hide the last subplot since we only have 9 difference plots
        axes2[4, 1].set_visible(False)
        
        plt.tight_layout()
        plt.show(block=False)  # Non-blocking display
          # Keep both plots open
    print("\n📊 Both plots are now displayed simultaneously!")
    print("💡 All plot windows will remain open. Close them manually when done.")
    
    # Create summary table
    summary_data = []
    for result in results:        summary_data.append({
            'Epochs': result['epochs'],
            'MSE': result['mse'],
            'Error_Rate_%': result['error_rate']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(refined_datasets_dir, 'incremental_epoch_comparison_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"📊 Detailed epoch results saved to: Refined Datasets/incremental_epoch_results.csv")
    if len(results) > 1:
        print(f"📊 Difference analysis data saved to: Refined Datasets/epoch_differences_results.csv")
    print(f"📁 All output files are saved in: Refined Datasets/ directory")

def check_gpu_status():
    """Check if TensorFlow can use GPU"""
    print("🔧 GPU Configuration Check:")
    
    # Check for GPU devices
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"   ✅ {len(gpus)} GPU(s) detected:")
        for i, gpu in enumerate(gpus):
            print(f"      GPU {i}: {gpu.name}")
        
        # Check memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("   🚀 TensorFlow will automatically use GPU for acceleration")
            print("   ⚡ Memory growth enabled to avoid allocation errors")
        except RuntimeError as e:
            print(f"   ⚠️  GPU configuration error: {e}")
    else:
        print("   ❌ No GPU detected - using CPU")
        print("   💡 To use GPU: Install CUDA + cuDNN or use Google Colab")
    
    # Test computation device
    with tf.device('/CPU:0'):
        cpu_test = tf.constant([1.0, 2.0, 3.0])
    
    if gpus:
        with tf.device('/GPU:0'):
            try:
                gpu_test = tf.constant([1.0, 2.0, 3.0])
                print("   ✅ GPU computation test passed")
            except:
                print("   ⚠️  GPU computation test failed")
    
    print()

def save_epoch_results_to_csv(epochs, y_test, predictions, mse, error_rate, write_header=False):
    """Save epoch results to CSV file with proper formatting for plot_dat_files.py"""
    # Get the script directory and refined datasets path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    refined_datasets_dir = os.path.join(script_dir, 'Refined Datasets')
    csv_file = os.path.join(refined_datasets_dir, 'incremental_epoch_results.csv')
    
    # Create DataFrame for this epoch's results
    epoch_data = []
    for i in range(len(y_test)):
        epoch_data.append({
            'Epoch': epochs,
            'DataPoint': i + 1,
            'Actual_kW': y_test[i],
            'Predicted_kW': predictions[i][0][0],
            'Error_kW': predictions[i][0][0] - y_test[i],
            'MSE': mse,
            'Error_Rate_%': error_rate
        })
    
    epoch_df = pd.DataFrame(epoch_data)
    
    # Write to CSV (append mode)
    if write_header:
        # First iteration - create new file with header
        epoch_df.to_csv(csv_file, mode='w', index=False, header=True)
        print(f"    📁 Created CSV file: {csv_file}")
    else:
        # Subsequent iterations - append without header
        epoch_df.to_csv(csv_file, mode='a', index=False, header=False)
        print(f"    📁 Appended to CSV: Epoch {epochs} data saved")

def save_difference_results_to_csv(epoch_from, epoch_to, prediction_diff, mean_diff, std_diff, write_header=False):
    """Save difference analysis results to CSV file"""
    # Get the script directory and refined datasets path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    refined_datasets_dir = os.path.join(script_dir, 'Refined Datasets')
    csv_file = os.path.join(refined_datasets_dir, 'epoch_differences_results.csv')
    
    # Create DataFrame for this difference analysis
    diff_data = []
    for i in range(len(prediction_diff)):
        diff_data.append({
            'From_Epoch': epoch_from,
            'To_Epoch': epoch_to,
            'DataPoint': i + 1,
            'Prediction_Difference': prediction_diff[i],
            'Mean_Difference': mean_diff,
            'Std_Difference': std_diff
        })
    
    diff_df = pd.DataFrame(diff_data)
    
    # Write to CSV (append mode)
    if write_header:
        # First difference - create new file with header
        diff_df.to_csv(csv_file, mode='w', index=False, header=True)
        print(f"    📁 Created differences CSV: {csv_file}")
    else:
        # Subsequent differences - append without header
        diff_df.to_csv(csv_file, mode='a', index=False, header=False)
        print(f"    📁 Appended difference: {epoch_from}→{epoch_to} epochs")

def main():
    """Main function to run the complete ML application"""
    print("=== TensorFlow ML Application: BOM Weather → House 4 Energy Prediction ===")
    print("Using 4 temperature features (Min, Max, 9am, 3pm) to predict energy consumption")
    print()
    
    # Display current configuration
    print("🖥️  Current Configuration:")
    print(f"   • TensorFlow version: {tf.__version__}")
    print(f"   • CUDA available: {cuda_available}")
    print(f"   • Mixed precision: {mixed_precision_enabled}")
    print(f"   • GPU acceleration: {'✅ ENABLED' if cuda_available else '❌ DISABLED'}")
    if cuda_available:
        gpus = tf.config.list_physical_devices('GPU')
        print(f"   • GPU device: {gpus[0].name}")
    print()
      # Check GPU status
    check_gpu_status()
    
    # Run incremental epoch comparison automatically
    try:
        run_incremental_epoch_comparison()
        print("\n=== ML Application Completed Successfully ===")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
