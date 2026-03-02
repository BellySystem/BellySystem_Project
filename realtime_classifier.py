#!/usr/bin/env python3
"""
Clasificador de Gestos MPU-6050 en Tiempo Real
"""

import joblib
import numpy as np
import pandas as pd
from collections import deque
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq
import time
import argparse
import warnings

from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

class RealtimeGestureClassifier:
    def __init__(self, model_path='gesture_model.pkl', 
                 window_size=2.0, stride=0.5, fs=50,
                 max_ip='127.0.0.1', max_port=9000):
        print(f"📦 Cargando modelo: {model_path}")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.class_names = model_data['class_names']
        
        print(f"✅ Modelo cargado - Clases: {', '.join(self.class_names)}")
        
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.window_samples = int(window_size * fs)
        self.stride_samples = int(stride * fs)
        
        max_buffer_size = int(fs * 5)
        self.acc_buffer = deque(maxlen=max_buffer_size)
        self.gyr_buffer = deque(maxlen=max_buffer_size)
        self.time_buffer = deque(maxlen=max_buffer_size)
        
        self.max_client = udp_client.SimpleUDPClient(max_ip, max_port)
        
        self.last_classification_time = 0
        self.sample_count = 0
        self.last_gesture = "ESPERANDO"
        self.gesture_confidence = 0.0
        
        self.classification_count = 0
        self.avg_classification_time = 0
        
        print(f"🎯 Enviando resultados a Max: {max_ip}:{max_port}")
        print(f"⚙️  Ventana: {window_size}s | Stride: {stride}s")
    
    def handle_acc(self, unused_addr, x, y, z):
        timestamp = time.time()
        self.acc_buffer.append([x, y, z])
        self.time_buffer.append(timestamp)
        self.sample_count += 1
        
        self.max_client.send_message("/acc/xyz", [x, y, z])
        self.try_classify()
    
    def handle_gyr(self, unused_addr, x, y, z):
        self.gyr_buffer.append([x, y, z])
        self.max_client.send_message("/gyr/xyz", [x, y, z])
    
    def try_classify(self):
        current_time = time.time()
        
        if len(self.acc_buffer) < self.window_samples:
            return
        
        if len(self.gyr_buffer) < self.window_samples:
            return
        
        if current_time - self.last_classification_time < self.stride:
            return
        
        self.classify_current_window()
        self.last_classification_time = current_time
    
    def classify_current_window(self):
        start_time = time.time()
        
        acc_window = np.array(list(self.acc_buffer)[-self.window_samples:])
        gyr_window = np.array(list(self.gyr_buffer)[-self.window_samples:])
        
        window_data = pd.DataFrame({
            'AccX': acc_window[:, 0],
            'AccY': acc_window[:, 1],
            'AccZ': acc_window[:, 2],
            'GyrX': gyr_window[:, 0],
            'GyrY': gyr_window[:, 1],
            'GyrZ': gyr_window[:, 2]
        })
        
        features = self.extract_features(window_data)
        feature_vector = [features[name] for name in self.feature_names]
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        features_scaled = self.scaler.transform(feature_vector)
        
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities[prediction]
        
        gesture_name = self.class_names[prediction]
        
        self.last_gesture = gesture_name
        self.gesture_confidence = confidence
        self.classification_count += 1
        
        classification_time = (time.time() - start_time) * 1000
        self.avg_classification_time = (self.avg_classification_time * 0.9 + 
                                       classification_time * 0.1)
        
        self.max_client.send_message("/gesture", gesture_name)
        self.max_client.send_message("/gesture/confidence", confidence)
        
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        for i, idx in enumerate(top3_indices):
            self.max_client.send_message(f"/gesture/top{i+1}", 
                                        [self.class_names[idx], float(probabilities[idx])])
        
        bar = "█" * int(confidence * 20)
        print(f"🎯 {gesture_name:15} │{bar:20}│ {confidence:.1%} │ "
              f"{classification_time:.1f}ms │ #{self.classification_count}")
    
    def extract_features(self, window_data):
        features = {}
        
        acc_mag = np.sqrt(window_data['AccX']**2 + window_data['AccY']**2 + window_data['AccZ']**2)
        gyr_mag = np.sqrt(window_data['GyrX']**2 + window_data['GyrY']**2 + window_data['GyrZ']**2)
        
        for axis in ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']:
            data = window_data[axis].values
            features[f'{axis}_mean'] = np.mean(data)
            features[f'{axis}_std'] = np.std(data)
            features[f'{axis}_min'] = np.min(data)
            features[f'{axis}_max'] = np.max(data)
            features[f'{axis}_range'] = features[f'{axis}_max'] - features[f'{axis}_min']
            features[f'{axis}_rms'] = np.sqrt(np.mean(data**2))
            features[f'{axis}_skew'] = stats.skew(data)
            features[f'{axis}_kurtosis'] = stats.kurtosis(data)
        
        features['acc_mag_mean'] = np.mean(acc_mag)
        features['acc_mag_std'] = np.std(acc_mag)
        features['acc_mag_max'] = np.max(acc_mag)
        features['acc_mag_rms'] = np.sqrt(np.mean(acc_mag**2))
        
        features['gyr_mag_mean'] = np.mean(gyr_mag)
        features['gyr_mag_std'] = np.std(gyr_mag)
        features['gyr_mag_max'] = np.max(gyr_mag)
        features['gyr_mag_rms'] = np.sqrt(np.mean(gyr_mag**2))
        
        features['acc_rise_time'] = self._calculate_rise_time(acc_mag)
        features['gyr_rise_time'] = self._calculate_rise_time(gyr_mag)
        
        peaks_acc, props_acc = signal.find_peaks(acc_mag, prominence=1.0)
        features['acc_num_peaks'] = len(peaks_acc)
        features['acc_peak_prominence'] = np.mean(props_acc['prominences']) if len(peaks_acc) > 0 else 0
        
        peaks_gyr, props_gyr = signal.find_peaks(gyr_mag, prominence=0.5)
        features['gyr_num_peaks'] = len(peaks_gyr)
        features['gyr_peak_prominence'] = np.mean(props_gyr['prominences']) if len(peaks_gyr) > 0 else 0
        
        features['acc_zero_crossings'] = self._count_zero_crossings(window_data['AccX'])
        features['gyr_zero_crossings'] = self._count_zero_crossings(window_data['GyrX'])
        
        freqs_acc, spectrum_acc = self._compute_spectrum(acc_mag)
        features['acc_spectral_centroid'] = self._spectral_centroid(freqs_acc, spectrum_acc)
        features['acc_spectral_energy'] = np.sum(spectrum_acc**2)
        features['acc_dominant_freq'] = freqs_acc[np.argmax(spectrum_acc)] if len(spectrum_acc) > 0 else 0
        
        freqs_gyr, spectrum_gyr = self._compute_spectrum(gyr_mag)
        features['gyr_spectral_centroid'] = self._spectral_centroid(freqs_gyr, spectrum_gyr)
        features['gyr_spectral_energy'] = np.sum(spectrum_gyr**2)
        features['gyr_dominant_freq'] = freqs_gyr[np.argmax(spectrum_gyr)] if len(spectrum_gyr) > 0 else 0
        
        features['acc_gyr_correlation'] = np.corrcoef(acc_mag, gyr_mag)[0, 1]
        
        half = len(window_data) // 2
        acc_mag_first_half = acc_mag[:half]
        features['acc_first_half_max'] = np.max(acc_mag_first_half)
        features['acc_first_half_energy'] = np.sum(acc_mag_first_half**2)
        
        return features
    
    def _calculate_rise_time(self, signal_data):
        peak = np.max(signal_data)
        if peak < 1.0:
            return 0
        idx_10 = np.argmax(signal_data > peak * 0.1)
        idx_90 = np.argmax(signal_data > peak * 0.9)
        if idx_90 <= idx_10:
            return 0
        return (idx_90 - idx_10) / self.fs * 1000
    
    def _count_zero_crossings(self, signal_data):
        return np.sum(np.diff(np.sign(signal_data)) != 0)
    
    def _compute_spectrum(self, signal_data):
        spectrum = np.abs(rfft(signal_data))
        freqs = rfftfreq(len(signal_data), 1/self.fs)
        return freqs, spectrum
    
    def _spectral_centroid(self, freqs, spectrum):
        return np.sum(freqs * spectrum) / np.sum(spectrum) if np.sum(spectrum) > 0 else 0


def main():
    parser = argparse.ArgumentParser(description='Clasificador de gestos en tiempo real')
    parser.add_argument('--model', default='gesture_model.pkl', help='Ruta al modelo')
    parser.add_argument('--esp32-port', type=int, default=8000, help='Puerto para recibir del ESP32')
    parser.add_argument('--max-ip', default='127.0.0.1', help='IP de Max/MSP')
    parser.add_argument('--max-port', type=int, default=9000, help='Puerto de Max/MSP')
    parser.add_argument('--window', type=float, default=2.0, help='Tamaño de ventana (segundos)')
    parser.add_argument('--stride', type=float, default=0.5, help='Stride (segundos)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CLASIFICADOR DE GESTOS MPU-6050 EN TIEMPO REAL")
    print("="*70)
    
    classifier = RealtimeGestureClassifier(
        model_path=args.model,
        window_size=args.window,
        stride=args.stride,
        max_ip=args.max_ip,
        max_port=args.max_port
    )
    
    dispatcher = Dispatcher()
    dispatcher.map("/acc/xyz", classifier.handle_acc)
    dispatcher.map("/gyr/xyz", classifier.handle_gyr)
    
    server = BlockingOSCUDPServer(("0.0.0.0", args.esp32_port), dispatcher)
    
    print(f"\n🎧 Escuchando ESP32 en puerto {args.esp32_port}")
    print(f"📡 Esperando datos del ESP32...")
    print(f"\n{'Gesto':<15} │ {'Confianza':<20} │ Tiempo │ Count")
    print("─" * 70)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Clasificador detenido")
        print(f"📊 Total clasificaciones: {classifier.classification_count}")
        print(f"⏱️  Tiempo promedio: {classifier.avg_classification_time:.1f}ms")


if __name__ == "__main__":
    main()

