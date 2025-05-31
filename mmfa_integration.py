#!/usr/bin/env python3
"""
Simplified Multi-Modal AI Fusion Accelerator (MMFA) Integration
Optimized for existing project structure with .tflite models and Verilog files
Author: MMFA Development Team
Version: 2.1 - Simplified
"""

import os
import sys
import time
import json
import numpy as np
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Try to import TensorFlow Lite with fallback
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        TFLITE_AVAILABLE = True
        tf = None  # Use tflite_runtime instead
    except ImportError:
        TFLITE_AVAILABLE = False

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class VerilogHardwareInterface:
    """Interface to existing Verilog hardware modules"""
    
    def __init__(self):
        self.verilog_dir = Path("verilog")
        self.hardware_ready = False
        
    def initialize_hardware(self) -> bool:
        """Initialize hardware modules"""
        required_files = [
            "dma_engine.v",
            "memory_controller.v", 
            "sram_buffer.v"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.verilog_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"{Colors.WARNING}⚠️ Missing Verilog files: {missing_files}{Colors.ENDC}")
            print(f"{Colors.WARNING}⚠️ Using simulation mode{Colors.ENDC}")
            return False
        
        # Check for Icarus Verilog
        try:
            result = subprocess.run(['iverilog', '-v'], capture_output=True, timeout=3)
            if result.returncode == 0:
                self.hardware_ready = True
                print(f"{Colors.OKGREEN}✅ Hardware simulation ready{Colors.ENDC}")
                return True
        except:
            pass
            
        print(f"{Colors.WARNING}⚠️ Icarus Verilog not found - using mathematical simulation{Colors.ENDC}")
        return False
    
    def dma_transfer(self, channel: str, data_size: int) -> Dict[str, float]:
        """Simulate DMA transfer with realistic parameters"""
        channel_priorities = {'vision': 3, 'audio': 2, 'motion': 1}
        priority = channel_priorities.get(channel, 1)
        
        if self.hardware_ready:
            # Could interface with actual Verilog simulation here
            base_latency = 45 + (4 - priority) * 15  # Priority affects latency
            transfer_rate = 850  # MB/s
        else:
            # Mathematical simulation
            base_latency = 50 + (4 - priority) * 20
            transfer_rate = 800  # MB/s
        
        transfer_time = (data_size * 8) / transfer_rate * 1000  # Convert to ns
        total_latency = base_latency + transfer_time
        power = 8.2 + (data_size / 1024) * 0.6
        
        return {
            'latency_ns': total_latency,
            'throughput_mbps': (data_size * 8) / (total_latency / 1000),
            'power_mw': power,
            'channel': channel,
            'priority': priority
        }
    
    def memory_access(self, operation: str, size: int) -> Dict[str, float]:
        """Simulate memory controller access"""
        if self.hardware_ready:
            # Could interface with actual memory controller
            cycles = np.random.randint(3, 7)
            power = 14.5 + np.random.uniform(-1.5, 1.5)
        else:
            cycles = np.random.randint(2, 6)
            power = 13.0 + np.random.uniform(-1.0, 1.0)
        
        return {
            'cycles': cycles,
            'latency_ns': cycles * 10,  # 100MHz clock
            'power_mw': power,
            'operation': operation
        }

class AIModelLoader:
    """Loads and manages TensorFlow Lite models"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models = {}
        self.model_configs = {
            'vision_model.tflite': 'vision',
            'audio_model.tflite': 'audio', 
            'motion_model.tflite': 'motion'
        }
        
    def load_all_models(self) -> Dict[str, bool]:
        """Load all available .tflite models"""
        results = {}
        
        for filename, model_name in self.model_configs.items():
            model_path = self.models_dir / filename
            success = self._load_single_model(model_path, model_name)
            results[model_name] = success
            
        return results
    
    def _load_single_model(self, model_path: Path, model_name: str) -> bool:
        """Load a single TensorFlow Lite model"""
        if not model_path.exists():
            print(f"{Colors.WARNING}⚠️ Model not found: {model_path}{Colors.ENDC}")
            return False
            
        if not TFLITE_AVAILABLE:
            print(f"{Colors.WARNING}⚠️ TensorFlow Lite not available for {model_name}{Colors.ENDC}")
            return False
            
        try:
            # Load with TensorFlow Lite or tflite_runtime
            if tf is not None:
                interpreter = tf.lite.Interpreter(model_path=str(model_path))
            else:
                interpreter = tflite.Interpreter(model_path=str(model_path))
                
            interpreter.allocate_tensors()
            
            # Get model details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            self.models[model_name] = {
                'interpreter': interpreter,
                'input_details': input_details,
                'output_details': output_details,
                'input_shape': input_details[0]['shape'],
                'output_shape': output_details[0]['shape']
            }
            
            print(f"{Colors.OKGREEN}✅ Loaded {model_name}: {model_path.name}{Colors.ENDC}")
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ Failed to load {model_name}: {e}{Colors.ENDC}")
            return False
    
    def run_inference(self, model_name: str, input_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run inference on specified model"""
        if model_name not in self.models:
            # Return mock prediction if model not loaded
            mock_output = np.random.random((1, 4)).astype(np.float32)
            mock_output = mock_output / np.sum(mock_output)  # Normalize
            return mock_output, np.random.uniform(8.0, 15.0)
        
        start_time = time.perf_counter()
        
        try:
            model_info = self.models[model_name]
            interpreter = model_info['interpreter']
            
            # Set input tensor
            interpreter.set_tensor(model_info['input_details'][0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output tensor
            output = interpreter.get_tensor(model_info['output_details'][0]['index'])
            
            inference_time = (time.perf_counter() - start_time) * 1000  # ms
            
            return output, inference_time
            
        except Exception as e:
            print(f"{Colors.WARNING}⚠️ Inference error for {model_name}: {e}{Colors.ENDC}")
            # Return mock prediction on error
            mock_output = np.random.random((1, 4)).astype(np.float32)
            return mock_output, 10.0

class SensorDataGenerator:
    """Generates realistic sensor data matching your model input requirements"""
    
    @staticmethod
    def generate_vision_frame():
        """Generate vision data - adjust shape based on your model"""
        # Common vision model input shapes
        return np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8).astype(np.float32) / 255.0
    
    @staticmethod
    def generate_audio_features():
        """Generate audio features - adjust based on your model"""
        # Common audio model shapes (MFCC, spectrograms, etc.)
        return np.random.normal(0, 1, (1, 128, 13)).astype(np.float32)
    
    @staticmethod
    def generate_motion_data():
        """Generate motion sensor data - adjust based on your model"""
        # IMU data: accelerometer + gyroscope
        accel = np.random.normal([0, 0, 9.8], 0.5, 3)  # Include gravity
        gyro = np.random.normal([0, 0, 0], 0.2, 3)
        return np.array([[*accel, *gyro]], dtype=np.float32)

class MultiModalFusionProcessor:
    """Handles multi-modal sensor fusion and decision making"""
    
    def __init__(self):
        self.fusion_weights = {'vision': 0.4, 'audio': 0.35, 'motion': 0.25}
        self.threat_threshold = 0.7
        self.decision_history = []
    
    def fuse_and_decide(self, predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Fuse predictions and make final decision"""
        threat_scores = {}
        
        # Extract threat probability from each modality
        for modality, prediction in predictions.items():
            if prediction is not None and len(prediction) > 0:
                # Assume last element or max value represents threat probability
                if prediction.shape[-1] > 1:
                    threat_scores[modality] = float(np.max(prediction))
                else:
                    threat_scores[modality] = float(prediction.flatten()[0])
        
        # Weighted fusion
        fused_score = 0.0
        total_weight = 0.0
        
        for modality, score in threat_scores.items():
            if modality in self.fusion_weights:
                fused_score += score * self.fusion_weights[modality]
                total_weight += self.fusion_weights[modality]
        
        if total_weight > 0:
            fused_score /= total_weight
        
        # Decision making
        is_threat = fused_score > self.threat_threshold
        confidence = fused_score if is_threat else (1.0 - fused_score)
        
        decision = {
            'timestamp': time.time(),
            'individual_scores': threat_scores,
            'fused_score': fused_score,
            'is_threat': is_threat,
            'confidence': confidence,
            'decision': '🚨 THREAT DETECTED' if is_threat else '✅ NORMAL'
        }
        
        self.decision_history.append(decision)
        return decision

class PerformanceTracker:
    """Tracks system performance metrics"""
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.metrics = {
            'frame_count': 0,
            'total_processing_time': 0.0,
            'total_inference_time': 0.0,
            'total_dma_time': 0.0,
            'total_power': 0.0,
            'threat_detections': 0,
            'frame_times': [],
            'power_samples': []
        }
    
    def log_frame(self, frame_data: Dict[str, Any]):
        """Log metrics for a single frame"""
        self.metrics['frame_count'] += 1
        self.metrics['total_processing_time'] += frame_data.get('processing_time', 0)
        self.metrics['total_inference_time'] += frame_data.get('inference_time', 0)
        self.metrics['total_dma_time'] += frame_data.get('dma_time', 0)
        self.metrics['total_power'] += frame_data.get('power_consumption', 0)
        
        if frame_data.get('threat_detected', False):
            self.metrics['threat_detections'] += 1
            
        self.metrics['frame_times'].append(frame_data.get('processing_time', 0))
        self.metrics['power_samples'].append(frame_data.get('power_consumption', 0))
    
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary statistics"""
        if self.metrics['frame_count'] == 0:
            return {}
        
        avg_frame_time = self.metrics['total_processing_time'] / self.metrics['frame_count']
        avg_fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
        avg_power = self.metrics['total_power'] / self.metrics['frame_count']
        threat_rate = self.metrics['threat_detections'] / self.metrics['frame_count']
        
        return {
            'total_frames': self.metrics['frame_count'],
            'avg_fps': avg_fps,
            'avg_frame_time_ms': avg_frame_time,
            'avg_inference_time_ms': self.metrics['total_inference_time'] / self.metrics['frame_count'],
            'avg_power_mw': avg_power,
            'peak_power_mw': max(self.metrics['power_samples']) if self.metrics['power_samples'] else 0,
            'threat_detection_rate': threat_rate,
            'efficiency_fps_per_watt': avg_fps / (avg_power / 1000.0) if avg_power > 0 else 0
        }

class SimplifiedMMFASystem:
    """Main MMFA system integrating all components"""
    
    def __init__(self):
        self.hardware = VerilogHardwareInterface()
        self.ai_models = AIModelLoader()
        self.data_generator = SensorDataGenerator()
        self.fusion_processor = MultiModalFusionProcessor()
        self.performance_tracker = PerformanceTracker()
        
        # Ensure directories exist
        Path("sim_results").mkdir(exist_ok=True)
    
    def initialize(self) -> bool:
        """Initialize the complete system"""
        print(f"\n{Colors.HEADER}{'='*50}")
        print("MMFA System Initialization")
        print(f"{'='*50}{Colors.ENDC}\n")
        
        # Initialize hardware
        hw_ready = self.hardware.initialize_hardware()
        
        # Load AI models
        model_status = self.ai_models.load_all_models()
        
        # Print status
        print(f"\n{Colors.BOLD}System Status:{Colors.ENDC}")
        print(f"  Hardware Simulation: {'✅ Ready' if hw_ready else '⚠️ Mock Mode'}")
        for model_name, loaded in model_status.items():
            status = '✅ Loaded' if loaded else '⚠️ Mock Mode'
            print(f"  {model_name.capitalize()} Model: {status}")
        
        print(f"\n{Colors.OKGREEN}✅ MMFA System Ready{Colors.ENDC}")
        return True
    
    def process_single_frame(self) -> Dict[str, Any]:
        """Process one frame of multi-modal data"""
        frame_start = time.perf_counter()
        
        # Generate sensor data
        vision_data = self.data_generator.generate_vision_frame()
        audio_data = self.data_generator.generate_audio_features()
        motion_data = self.data_generator.generate_motion_data()
        
        # Simulate DMA transfers
        dma_results = {}
        total_dma_time = 0
        for modality, data in [('vision', vision_data), ('audio', audio_data), ('motion', motion_data)]:
            dma_result = self.hardware.dma_transfer(modality, data.nbytes)
            dma_results[modality] = dma_result
            total_dma_time += dma_result['latency_ns']
        
        # AI inference
        inference_start = time.perf_counter()
        predictions = {}
        
        vision_pred, vision_time = self.ai_models.run_inference('vision', vision_data)
        audio_pred, audio_time = self.ai_models.run_inference('audio', audio_data)
        motion_pred, motion_time = self.ai_models.run_inference('motion', motion_data)
        
        predictions = {'vision': vision_pred, 'audio': audio_pred, 'motion': motion_pred}
        total_inference_time = vision_time + audio_time + motion_time
        
        # Multi-modal fusion
        decision = self.fusion_processor.fuse_and_decide(predictions)
        
        # Memory access simulation
        memory_result = self.hardware.memory_access('write', sum(data.nbytes for data in [vision_data, audio_data, motion_data]))
        
        # Calculate metrics
        processing_time = (time.perf_counter() - frame_start) * 1000  # ms
        power_consumption = sum(result['power_mw'] for result in dma_results.values()) + memory_result['power_mw'] + 35.0  # Base system power
        
        frame_data = {
            'processing_time': processing_time,
            'inference_time': total_inference_time,
            'dma_time': total_dma_time / 1_000_000,  # Convert ns to ms
            'power_consumption': power_consumption,
            'threat_detected': decision['is_threat'],
            'decision': decision,
            'dma_results': dma_results,
            'memory_result': memory_result
        }
        
        # Log performance
        self.performance_tracker.log_frame(frame_data)
        
        return frame_data
    
    def run_simulation(self, num_frames: int = 100, verbose: bool = True) -> Dict[str, Any]:
        """Run complete simulation"""
        print(f"\n{Colors.BOLD}Running MMFA Simulation - {num_frames} frames{Colors.ENDC}")
        print("-" * 50)
        
        self.performance_tracker.reset_metrics()
        
        for frame_idx in range(num_frames):
            frame_data = self.process_single_frame()
            
            if verbose and frame_idx % 20 == 0:
                decision = frame_data['decision']
                fps = 1000.0 / frame_data['processing_time'] if frame_data['processing_time'] > 0 else 0
                print(f"Frame {frame_idx:3d}: {decision['decision'][:12]:12} | "
                      f"FPS: {fps:5.1f} | Power: {frame_data['power_consumption']:5.1f}mW | "
                      f"Conf: {decision['confidence']:.3f}")
        
        # Get final results
        performance_summary = self.performance_tracker.get_summary()
        
        # Save results
        results = {
            'simulation_config': {
                'num_frames': num_frames,
                'timestamp': time.time()
            },
            'performance_summary': performance_summary,
            'system_config': {
                'models_loaded': list(self.ai_models.models.keys()),
                'hardware_simulation': self.hardware.hardware_ready
            }
        }
        
        # Save to file
        results_file = Path("sim_results") / f"mmfa_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{Colors.OKGREEN}✅ Simulation completed{Colors.ENDC}")
        print(f"📊 Results saved: {results_file}")
        
        return results
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print formatted results summary"""
        summary = results['performance_summary']
        
        print(f"\n{Colors.HEADER}{'='*50}")
        print("MMFA Performance Summary")
        print(f"{'='*50}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}Processing Performance:{Colors.ENDC}")
        print(f"  Frames Processed: {summary['total_frames']}")
        print(f"  Average FPS: {summary['avg_fps']:.1f}")
        print(f"  Average Frame Time: {summary['avg_frame_time_ms']:.2f} ms")
        print(f"  Average Inference Time: {summary['avg_inference_time_ms']:.2f} ms")
        
        print(f"\n{Colors.BOLD}Power & Efficiency:{Colors.ENDC}")
        print(f"  Average Power: {summary['avg_power_mw']:.1f} mW")
        print(f"  Peak Power: {summary['peak_power_mw']:.1f} mW")
        print(f"  Efficiency: {summary['efficiency_fps_per_watt']:.1f} FPS/W")
        
        print(f"\n{Colors.BOLD}AI Detection:{Colors.ENDC}")
        print(f"  Threat Detection Rate: {summary['threat_detection_rate']:.1%}")
        print(f"  Total Threats Detected: {int(summary['total_frames'] * summary['threat_detection_rate'])}")

def main():
    """Main execution function"""
    try:
        # Create and initialize MMFA system
        mmfa_system = SimplifiedMMFASystem()
        
        if not mmfa_system.initialize():
            print(f"{Colors.FAIL}❌ System initialization failed{Colors.ENDC}")
            return 1
        
        # Run simulation
        results = mmfa_system.run_simulation(num_frames=50, verbose=True)
        
        # Print results
        mmfa_system.print_results_summary(results)
        
        print(f"\n{Colors.OKGREEN}🎯 MMFA simulation completed successfully!{Colors.ENDC}")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠️ Simulation interrupted{Colors.ENDC}")
        return 0
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

