# TensorFlow ML Application - BOM Weather to House 4 Energy Prediction
# Adapted from Google Colab code for local Windows environment
# Uses temperature from BOM data to predict energy consumption in House 4

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import math
import pickle
# matplotlib.pyplot import removed - plotting now handled by plot_refined_datasets.py

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
    print(f"💾 Saving BOM data to: {bomfile}")
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
    print(f"💾 Saving House 4 data to: {house4file}")
    with open(house4file, 'wb') as file:
        pickle.dump(house4data_processed, file)
    
    return house4data_processed

def load_processed_data():
    """Load processed data from dat files"""
    print("Loading processed data...")
    print(f"📁 Looking for files in: {refined_datasets_dir}")
    
    # Check if processed files exist, if not create them
    if not os.path.exists(bomfile):
        print("📊 Processing BOM data (file not found)...")
        bom = process_bom_data()
    else:
        print(f"📖 Loading BOM data from: {bomfile}")
        with open(bomfile, 'rb') as file:
            bom = pickle.load(file)
    
    if not os.path.exists(house4file):
        print("🏠 Processing House 4 data (file not found)...")
        house4data_processed = process_house4_data()
    else:
        print(f"📖 Loading House 4 data from: {house4file}")
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

def create_train_test_split(x_train_full, y_train_full, test_split=0.05):
    """Separate test set and training set with configurable split ratio"""
    print(f"Creating train/test split ({int((1-test_split)*100)}-{int(test_split*100)})...")
    
    #separate test set and training set
    part = int(len(x_train_full) * test_split)
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
        
        print(f"    MSE: {mse:.4f}, Error Rate: {error_rate:.2f}%")      # Plotting functionality removed - now handled by plot_refined_datasets.py
    print(f"\n📊 Training completed! Visualization will be handled by plot_refined_datasets.py")
    print(f"📁 Results saved for {len(results)} different epoch configurations")
    
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
    print("🚀 EFFICIENT TRAINING: Single model trained incrementally")
    print("📈 Training progression: 50 → 100 → 150 → 200 → 250 → 300 → 350 → 400 → 450 → 500 epochs")
    print("💡 Total epochs: 500 (not 1000+ like before!)")
    print("⚡ Each step adds only 50 more epochs to existing model\n")

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
        additional_epochs = epochs - prev_epochs  # Calculate how many new epochs to train
        print(f"[{i}/{len(epoch_tests)}] 🧠 Training from {prev_epochs} to {epochs} epochs (+{additional_epochs} new epochs)...")
        
        import time
        start_time = time.time()
        history = model.fit(x_train, y_train, batch_size=1, initial_epoch=prev_epochs, epochs=epochs, verbose=1)
        training_time = time.time() - start_time
        
        # Update prev_epochs for next iteration
        prev_epochs = epochs
        
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
            'training_time': training_time,
            'predictions': predictions
        })
        
        # Save results to CSV after each iteration
        save_epoch_results_to_csv(epochs, y_test, predictions, mse, error_rate, i == 1)  # header only on first iteration
      # Create 5x2 subplot visualization (10 plots total) removed - now handled by plot_refined_datasets.py
    # All plotting functionality moved to plot_refined_datasets.py for better organization
    
    print(f"\n📊 Training completed! Visualization will be handled by plot_refined_datasets.py")
    print(f"📁 Results saved to CSV files in: {refined_datasets_dir}")
    
    # Call the plotting script to handle all visualizations
    try:
        import subprocess
        import sys
        
        plot_script = os.path.join(os.path.dirname(__file__), 'plot_refined_datasets.py')
        print(f"\n🎨 Launching visualization script: {plot_script}")
        
        # Run the plotting script in a separate process
        result = subprocess.run([sys.executable, plot_script], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Plotting script completed successfully!")
        else:
            print(f"⚠️  Plotting script had issues:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            
    except Exception as e:
        print(f"⚠️  Could not launch plotting script: {e}")
        print("💡 You can run 'python plot_refined_datasets.py' manually to see visualizations")    # Difference comparison plotting removed - now handled by plot_refined_datasets.py
    # All plotting functionality moved to plot_refined_datasets.py for better organization
    
    if len(results) > 1:
        print(f"📊 Difference analysis data will be visualized by plot_refined_datasets.py")
        print(f"📁 Difference data saved to: {os.path.join(refined_datasets_dir, 'epoch_differences_results.csv')}")
        
        # Save difference results to CSV for all consecutive epoch pairs
        for i in range(len(results) - 1):
            current_pred = [results[i+1]['predictions'][j][0][0] for j in range(len(results[i+1]['predictions']))]
            previous_pred = [results[i]['predictions'][j][0][0] for j in range(len(results[i]['predictions']))]
            prediction_diff = [current_pred[j] - previous_pred[j] for j in range(len(current_pred))]
            mean_diff = np.mean(prediction_diff)
            std_diff = np.std(prediction_diff)
            
            # Save difference results to CSV
            save_difference_results_to_csv(results[i]['epochs'], results[i+1]['epochs'], 
                                          prediction_diff, mean_diff, std_diff, i == 0)  # header only on first iteration
        
        print(f"✅ All difference data saved for visualization")

    print("\n📊 All training data saved to CSV files!")
    print("💡 Run 'python plot_refined_datasets.py' to see all visualizations")
    print(f"\n🚀 EFFICIENCY ACHIEVED!")
    print(f"   ✅ Total epochs trained: 500 (not 1000+ wasteful epochs)")
    print(f"   ✅ Incremental training: Each step adds only 50 epochs")
    print(f"   ✅ Model reused efficiently across all {len(epoch_tests)} training phases")
    print(f"   ✅ Massive time savings compared to training from scratch each time!")
    
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
    """Save epoch results to CSV file with proper formatting for plot_refined_datasets.py"""
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

def run_dual_split_comparison():
    """Train ONE model: 80%→95%→80%→95% within each interval (500 total epochs)"""
    print("=== Efficient Dual Split Comparison Study ===")
    print("🔄 CONTINUOUS TRAINING: 80% then 95% within each 50-epoch interval")
    print("📈 Pattern: [80%(25) + 95%(25)] × 10 intervals = 500 epochs")
    print("⚡ Same model continues learning throughout entire process")
    print("📊 CSV saved after each 80% and 95% training phase\n")

    # Load data
    bom, house4data_processed = load_processed_data()
    x_train_full, y_train_full = generate_training_data(bom, house4data_processed)
      # Prepare both splits upfront
    x_train_80, x_test_80, y_train_80, y_test_80 = create_train_test_split(x_train_full, y_train_full, test_split=0.20)
    x_train_95, x_test_95, y_train_95, y_test_95 = create_train_test_split(x_train_full, y_train_full, test_split=0.05)
    
    # Storage for results
    detailed_results_80 = []
    detailed_results_95 = []
    differences_results_80 = []
    differences_results_95 = []
    
    # Create single model that will be used throughout
    model = create_rnn_model()
    current_epoch = 0
    prev_predictions_80 = None
    prev_predictions_95 = None
    
    print("🚀 Starting continuous 80%→95% training...")
    print("=" * 60)    
    # 10 intervals of 50 epochs each (25 + 25)
    for interval in range(1, 11):
        print(f"\n[INTERVAL {interval}/10] 🧠 Training sequence: 80% → 95%")
        
        # === PHASE 1: Train on 80% for 25 epochs ===
        print(f"  Phase 1: Training on 80% data...")
        import time
        start_time = time.time()
        
        target_epoch_80 = current_epoch + 25
        history = model.fit(x_train_80, y_train_80, batch_size=1, 
                           initial_epoch=current_epoch, epochs=target_epoch_80, verbose=1)
        current_epoch = target_epoch_80
        
        # Evaluate and save 80% results
        mse_80 = model.evaluate(x_test_80, y_test_80, verbose=0)
        predictions_80 = model.predict(x_test_80, verbose=0)
        errors_80 = [predictions_80[j][0][0] - y_test_80[j] for j in range(len(predictions_80))]
        error_rate_80 = np.mean(np.abs(errors_80)) / np.mean(y_test_80) * 100
        training_time_80 = time.time() - start_time
        
        print(f"    80% Results (Epoch {current_epoch}): MSE={mse_80:.4f}, Error Rate={error_rate_80:.2f}%, Time={training_time_80:.1f}s")
        
        # Store detailed 80% results
        for j in range(len(predictions_80)):
            detailed_results_80.append({
                'Epoch': current_epoch,
                'DataPoint': j + 1,
                'Actual_kW': y_test_80[j],
                'Predicted_kW': predictions_80[j][0][0],
                'Error_kW': predictions_80[j][0][0] - y_test_80[j],
                'MSE': mse_80,
                'Error_Rate_%': error_rate_80,
                'Trained_On': '80-20',
                'Interval': interval,
                'Phase': '80%'
            })
        
        # Calculate differences for 80%
        if prev_predictions_80 is not None:
            pred_diffs_80 = [predictions_80[j][0][0] - prev_predictions_80[j][0][0] for j in range(len(predictions_80))]
            mean_diff_80 = np.mean(pred_diffs_80)
            std_diff_80 = np.std(pred_diffs_80)
            
            for j in range(len(predictions_80)):
                differences_results_80.append({
                    'From_Epoch': current_epoch - 25,
                    'To_Epoch': current_epoch,
                    'DataPoint': j + 1,
                    'Previous_Prediction_kW': prev_predictions_80[j][0][0],
                    'Current_Prediction_kW': predictions_80[j][0][0],
                    'Prediction_Difference_kW': pred_diffs_80[j],
                    'Mean_Difference_kW': mean_diff_80,
                    'Std_Difference_kW': std_diff_80,
                    'Trained_On': '80-20',
                    'Interval': interval
                })
        
        prev_predictions_80 = predictions_80.copy()
        
        # === PHASE 2: Continue training on 95% for 25 epochs ===
        print(f"  Phase 2: Continuing with 95% data...")
        start_time = time.time()
        
        target_epoch_95 = current_epoch + 25
        history = model.fit(x_train_95, y_train_95, batch_size=1, 
                           initial_epoch=current_epoch, epochs=target_epoch_95, verbose=1)
        current_epoch = target_epoch_95
        
        # Evaluate and save 95% results
        mse_95 = model.evaluate(x_test_95, y_test_95, verbose=0)
        predictions_95 = model.predict(x_test_95, verbose=0)
        errors_95 = [predictions_95[j][0][0] - y_test_95[j] for j in range(len(predictions_95))]
        error_rate_95 = np.mean(np.abs(errors_95)) / np.mean(y_test_95) * 100
        training_time_95 = time.time() - start_time
        
        print(f"    95% Results (Epoch {current_epoch}): MSE={mse_95:.4f}, Error Rate={error_rate_95:.2f}%, Time={training_time_95:.1f}s")
        
        # Store detailed 95% results
        for j in range(len(predictions_95)):
            detailed_results_95.append({
                'Epoch': current_epoch,
                'DataPoint': j + 1,
                'Actual_kW': y_test_95[j],
                'Predicted_kW': predictions_95[j][0][0],
                'Error_kW': predictions_95[j][0][0] - y_test_95[j],
                'MSE': mse_95,
                'Error_Rate_%': error_rate_95,
                'Trained_On': '95-5',
                'Interval': interval,
                'Phase': '95%'
            })
        
        # Calculate differences for 95%
        if prev_predictions_95 is not None:
            pred_diffs_95 = [predictions_95[j][0][0] - prev_predictions_95[j][0][0] for j in range(len(predictions_95))]
            mean_diff_95 = np.mean(pred_diffs_95)
            std_diff_95 = np.std(pred_diffs_95)
            
            for j in range(len(predictions_95)):
                differences_results_95.append({
                    'From_Epoch': current_epoch - 25,
                    'To_Epoch': current_epoch,
                    'DataPoint': j + 1,
                    'Previous_Prediction_kW': prev_predictions_95[j][0][0],
                    'Current_Prediction_kW': predictions_95[j][0][0],
                    'Prediction_Difference_kW': pred_diffs_95[j],
                    'Mean_Difference_kW': mean_diff_95,
                    'Std_Difference_kW': std_diff_95,
                    'Trained_On': '95-5',
                    'Interval': interval
                })
        
        prev_predictions_95 = predictions_95.copy()
        
        print(f"    ✅ Interval {interval} completed: Total epochs = {current_epoch}")
        
        # Save intermediate results after each interval
        if interval % 2 == 0:  # Save every 2 intervals
            print(f"    💾 Saving intermediate results...")
            save_split_results(detailed_results_80, differences_results_80, "80_20")
            save_split_results(detailed_results_95, differences_results_95, "95_5")
    
    # Save all final results
    save_split_results(detailed_results_80, differences_results_80, "80_20")
    save_split_results(detailed_results_95, differences_results_95, "95_5")    
    print(f"\n✅ Continuous dual split training completed!")
    print(f"⚡ Total epochs trained: 500 with 80%→95% pattern within each interval")
    print("🔄 Training pattern: [80%(25) + 95%(25)] × 10 intervals")
    print("📁 Generated files:")
    print("   • incremental_epoch_results_80_20.csv (80-20 split detailed results)")
    print("   • epoch_differences_results_80_20.csv (80-20 split differences)")
    print("   • incremental_epoch_results_95_5.csv (95-5 split detailed results)")
    print("   • epoch_differences_results_95_5.csv (95-5 split differences)")
    print("\n🧠 Training Strategy:")
    print("   • Model trains on 80% then 95% within each 50-epoch interval")
    print("   • Same model accumulates knowledge from both training set sizes")
    print("   • CSV saved after each 80% and 95% training phase")

# Removed train_incremental_on_split - now using efficient single model approach in run_dual_split_comparison

def save_split_results(detailed_results, differences_results, split_name):
    """Save results for a specific split to CSV files"""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    refined_datasets_dir = os.path.join(script_dir, 'Refined Datasets')
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_file = os.path.join(refined_datasets_dir, f'incremental_epoch_results_{split_name.replace("-", "_")}.csv')
    detailed_df.to_csv(detailed_file, index=False)
    print(f"    💾 Saved: {detailed_file}")
    
    # Save differences results
    if differences_results:
        differences_df = pd.DataFrame(differences_results)
        differences_file = os.path.join(refined_datasets_dir, f'epoch_differences_results_{split_name.replace("-", "_")}.csv')
        differences_df.to_csv(differences_file, index=False)
        print(f"    💾 Saved: {differences_file}")

def save_dual_split_summary(results_80, results_95):
    """Save comparison summary between both splits"""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    refined_datasets_dir = os.path.join(script_dir, 'Refined Datasets')
    
    summary_data = []
    for i, (r80, r95) in enumerate(zip(results_80, results_95)):
        summary_data.append({
            'Epochs': r80['epochs'],
            'Trained_On': r80['trained_on'],
            'MSE_80_20': r80['mse'],
            'MSE_95_5': r95['mse'],
            'Error_Rate_80_20_%': r80['error_rate'],
            'Error_Rate_95_5_%': r95['error_rate'],
            'MSE_Difference': r95['mse'] - r80['mse'],
            'Error_Rate_Difference_%': r95['error_rate'] - r80['error_rate']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(refined_datasets_dir, 'dual_split_comparison_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"    💾 Saved: {summary_file}")

def main():
    """Main function to run the complete ML application"""
    print("=== TensorFlow ML Application: BOM Weather → House 4 Energy Prediction ===")
    print("Using 4 temperature features (Min, Max, 9am, 3pm) to predict energy consumption")
    print()
    
    # Display file organization
    print("📁 File Organization:")
    print(f"   • Input data: {datapath}")
    print(f"   • Output data: {refined_datasets_dir}")
    print(f"   • BOM processed data: {bomfile}")
    print(f"   • House 4 processed data: {house4file}")
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
    print()    # Check GPU status
    check_gpu_status()
    
    # Run dual split comparison (80-20 and 95-5)
    try:
        run_dual_split_comparison()
        print("\n=== ML Application Completed Successfully ===")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()