#!/usr/bin/env python3
"""
Motion Model Generator for Multi-Modal AI Fusion Accelerator
Generates motion_model.tflite for motion pattern detection using MLP
"""

import tensorflow as tf
import numpy as np
import os

def create_simple_motion_model():
    """Lightweight MLP for motion pattern detection - optimized for FPGA simulation"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),  # 6 features: accel_x,y,z + gyro_x,y,z
        
        # Simple dense layers for motion classification
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 classes: normal, fall, struggle, run
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_and_convert_motion():
    print("Creating Simple Motion Model for Vivado Simulation...")
    
    # Create model
    model = create_simple_motion_model() 
    model.summary()
    
    # Generate minimal dummy data (IMU sensor data)
    x_dummy = np.random.random((100, 6)).astype(np.float32) * 2 - 1  # Range [-1, 1]
    y_dummy = np.random.randint(4, size=(100,)).astype(np.int32)
    
    # Quick training
    print("Quick training for weight initialization...")
    model.fit(x_dummy, y_dummy, epochs=3, batch_size=20, verbose=1)
    
    # Convert to TFLite with quantization
    print("Converting to TFLite with INT8 quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # INT8 quantization
    def representative_dataset():
        for _ in range(20):
            yield [np.random.random((1, 6)).astype(np.float32) * 2 - 1]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'motion_model.tflite')
    
    with open(filepath, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Motion model saved: {filepath} ({len(tflite_model)} bytes)")
    return tflite_model

if __name__ == "__main__":
    generate_and_convert_motion()