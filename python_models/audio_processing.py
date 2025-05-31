#!/usr/bin/env python3
"""
Fixed Audio Model - TFLite Compatible
Replaces LSTM with 1D CNN for better TFLite compatibility
"""

import tensorflow as tf
import numpy as np
import os

def create_simple_audio_model():
    """Lightweight 1D CNN for audio threat detection - TFLite compatible"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 13)),  # 32 time steps, 13 MFCC features
        
        # Use 1D convolutions instead of LSTM for TFLite compatibility
        tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalMaxPooling1D(),
        
        # Dense layers for classification
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: normal, scream, crash
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_and_convert_audio():
    print("Creating TFLite-Compatible Audio Model...")
    
    # Create model
    model = create_simple_audio_model()
    model.summary()
    
    # Generate minimal dummy data
    x_dummy = np.random.random((50, 32, 13)).astype(np.float32)
    y_dummy = np.random.randint(3, size=(50,)).astype(np.int32)
    
    # Quick training
    print("Quick training for weight initialization...")
    model.fit(x_dummy, y_dummy, epochs=2, batch_size=10, verbose=1)
    
    # Convert to TFLite with quantization - FIXED VERSION
    print("Converting to TFLite with INT8 quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # INT8 quantization with proper configuration
    def representative_dataset():
        for _ in range(10):
            yield [np.random.random((1, 32, 13)).astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    
    # Use TFLite built-ins only (no SELECT_TF_OPS needed)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Add this to handle any remaining compatibility issues
    converter.allow_custom_ops = False
    converter.experimental_new_converter = True
    
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"⚠️  INT8 conversion failed, trying float16 fallback...")
        print(f"Error: {e}")
        
        # Fallback to float16 quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        print("✅ Using float16 quantization instead")
    
    # Create output directory if it doesn't exist
    output_dir = 'models'  # Save directly in the current working directory's output folder
    os.makedirs(output_dir, exist_ok=True)
    
    # Save TFLite model
    output_path = os.path.join(output_dir, 'audio_model.tflite')  # Correct path
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Audio model saved: {output_path} ({len(tflite_model)} bytes)")
    return tflite_model

if __name__ == "__main__":
    generate_and_convert_audio()