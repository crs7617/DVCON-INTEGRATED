#!/usr/bin/env python3
"""
Vision Model Generator for Multi-Modal AI Fusion Accelerator
Generates vision_model.tflite for threat detection using CNN
"""

import tensorflow as tf
import numpy as np
import os

def create_simple_vision_model():
    """Lightweight CNN for threat detection - optimized for FPGA simulation"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),  # Smaller input for simulation
        
        # Simple conv layers
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Simple dense layers
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary: threat/safe
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def generate_and_convert_vision():
    print("Creating Simple Vision Model for Vivado Simulation...")
    
    # Create model
    model = create_simple_vision_model()
    model.summary()
    
    # Generate minimal dummy data for quick training
    x_dummy = np.random.random((50, 64, 64, 3)).astype(np.float32)
    y_dummy = np.random.randint(2, size=(50, 1)).astype(np.float32)
    
    # Quick training (just to initialize weights properly)
    print("Quick training for weight initialization...")
    model.fit(x_dummy, y_dummy, epochs=2, batch_size=10, verbose=1)
    
    # Convert to TFLite with quantization
    print("Converting to TFLite with INT8 quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # INT8 quantization for maximum power efficiency
    def representative_dataset():
        for _ in range(10):
            yield [np.random.random((1, 64, 64, 3)).astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'vision_model.tflite')
    
    with open(filepath, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Vision model saved: {filepath} ({len(tflite_model)} bytes)")
    return tflite_model

if __name__ == "__main__":
    generate_and_convert_vision()

