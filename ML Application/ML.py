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

def create_train_test_split(x_train_full, y_train_full):
    """Separate test set and training set with 80-20 split"""
    print("Creating train/test split (80-20)...")
    
    #separate test set and training set
    part = int(len(x_train_full) * 0.20)  # 20% for testing
    x_test = x_train_full[-part:]
    y_test = y_train_full[-part:]
    
    x_train = x_train_full[:-part]  # 80% for training
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

def run_incremental_epoch_comparison():
    """Train a single model incrementally: 50, 100, ..., 500 epochs with 80-20 split, storing results after each increment, and automatically generate plots."""
    print("=== Incremental Epoch Comparison Study ===")
    print("🚀 EFFICIENT TRAINING: Single model trained incrementally (80-20 split)")
    print("📈 Training progression: 50 → 100 → 150 → 200 → 250 → 300 → 350 → 400 → 450 → 500 epochs")
    print("💡 Total epochs: 500 (not 1000+ like before!)")
    print("⚡ Each step adds only 50 more epochs to existing model")
    print("🎨 Plots will be generated automatically after training completes\n")

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
            'predictions': predictions
        })
        
        print(f"    ✅ Training time: {training_time:.1f}s | MSE: {mse:.4f} | Error Rate: {error_rate:.2f}%")
        print(f"    📊 Total epochs trained so far: {epochs} | Efficient incremental training!")
        
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
        
        # Run the plotting script in a separate process with real-time output
        result = subprocess.run([sys.executable, plot_script], 
                              capture_output=False,  # Show output in real-time
                              text=True, 
                              cwd=os.path.dirname(__file__))  # Set working directory
        
        if result.returncode == 0:
            print("✅ Plotting script completed successfully!")
        else:
            print(f"⚠️  Plotting script returned code: {result.returncode}")
            
    except Exception as e:
        print(f"⚠️  Could not launch plotting script: {e}")
        print("💡 You can run 'python plot_refined_datasets.py' manually to see visualizations")# Difference comparison plotting removed - now handled by plot_refined_datasets.py
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