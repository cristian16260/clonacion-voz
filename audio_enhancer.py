#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_enhancer.py
-----------------
Pipeline de Acondicionamiento y Limpieza Acústica Inteligente para Clonación de Voz.
- Filtro Paso-Banda Butterworth (75 Hz - 11.5 kHz) para eliminar rumble, corriente y siseos.
- Reducción Adaptativa de Ruido Espectral (STFT Spectral Gating / Wiener suave).
- Recorte inteligente de silencios (VAD Trim con preservación de ataques/colas).
- Normalización de Sonoridad RMS (-16 dBFS) con limitador anti-clipping (-0.5 dBFS).
- Remuestreo estándar a 24 kHz optimizado para XTTS-v2.
"""

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import numpy as np
import soundfile as sf
import scipy.signal as signal
import librosa

# Forzar UTF-8 en Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


@dataclass
class AudioEnhancementConfig:
    """Parámetros de configuración del pipeline acústico."""
    high_pass_hz: float = 75.0          # Elimina rumble, DC offset y vibraciones mecánicas
    low_pass_hz: float = 11500.0        # Elimina siseos agudos fuera del rango vocal útil
    noise_reduction_strength: float = 0.80  # Factor de atenuación de ruido (0.0 a 1.0)
    target_dbfs: float = -16.0          # Sonoridad objetivo RMS (estándar broadcast/podcast)
    trim_silence: bool = True           # Recortar silencios muertos al inicio y final
    trim_top_db: float = 28.0           # Umbral relativo de silencio en dB
    target_sr: int = 24000              # Frecuencia de muestreo nativa de XTTS-v2


class AudioEnhancer:
    """Procesador acústico puro de alta fidelidad y baja latencia."""

    @staticmethod
    def _apply_bandpass_filter(y: np.ndarray, sr: int, lowcut: float = 75.0, highcut: float = 11500.0) -> np.ndarray:
        """Aplica filtros Butterworth de fase lineal (forward-backward) para aislar frecuencias vocales."""
        nyquist = 0.5 * sr
        
        # Filtro Pasa-Altas (High-pass)
        if lowcut > 0 and lowcut < nyquist:
            sos_hp = signal.butter(4, lowcut / nyquist, btype="highpass", output="sos")
            y = signal.sosfiltfilt(sos_hp, y)
            
        # Filtro Pasa-Bajas (Low-pass)
        if highcut > 0 and highcut < nyquist:
            sos_lp = signal.butter(4, highcut / nyquist, btype="lowpass", output="sos")
            y = signal.sosfiltfilt(sos_lp, y)
            
        return y

    @staticmethod
    def _spectral_denoise(y: np.ndarray, sr: int, strength: float = 0.80) -> np.ndarray:
        """
        Reducción adaptativa de ruido espectral (STFT Soft Spectral Gating).
        Estima el piso de ruido estático a partir de los percentiles bajos de energía
        y aplica una ganancia suave que preserva los armónicos y formantes vocales sin
        generar artefactos metálicos (musical noise).
        """
        if len(y) < 512:
            return y
            
        n_fft = 1024
        hop_length = 256
        
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        mag, phase = np.abs(D), np.angle(D)
        
        # Estimar perfil de ruido: percentil 15 de cada bin de frecuencia a lo largo del tiempo
        noise_profile = np.percentile(mag, 15, axis=1, keepdims=True)
        
        # Matriz de relación señal-ruido estimada
        snr_matrix = mag / (noise_profile + 1e-8)
        
        # Máscara de ganancia suave (Wiener sigmoid)
        # Transición suave para atenuar ruido sin comerse los ataques fonéticos
        gain_mask = 1.0 / (1.0 + np.exp(-1.8 * (snr_matrix - (1.2 + strength))))
        
        # Atenuación mínima para evitar silencios artificiales
        min_gain = 1.0 - (0.90 * strength)
        gain_mask = np.maximum(gain_mask, min_gain)
        
        # Suavizado temporal y frecuencial de la máscara (filtro gaussiano / convolución leve)
        gain_mask = signal.medfilt2d(gain_mask, kernel_size=(3, 3))
        
        # Aplicar máscara y reconstruir señal
        clean_mag = mag * gain_mask
        clean_D = clean_mag * np.exp(1j * phase)
        clean_y = librosa.istft(clean_D, hop_length=hop_length, length=len(y))
        
        return clean_y

    @staticmethod
    def _trim_silences(y: np.ndarray, sr: int, top_db: float = 28.0) -> np.ndarray:
        """Recorta silencios iniciales y finales, agregando un pequeño margen (50ms) para proteger consonantes."""
        try:
            trimmed, index = librosa.effects.trim(y, top_db=top_db, frame_length=1024, hop_length=256)
            # Si el recorte dejó la señal vacía, devolver la original
            if len(trimmed) > int(0.2 * sr):
                pad_len = int(0.05 * sr) # 50ms padding
                start = max(0, index[0] - pad_len)
                end = min(len(y), index[1] + pad_len)
                return y[start:end]
        except Exception:
            pass
        return y

    @staticmethod
    def _normalize_loudness(y: np.ndarray, target_dbfs: float = -16.0) -> np.ndarray:
        """Normaliza la sonoridad RMS al estándar objetivo y limita picos a -0.5 dBFS para evitar clipping."""
        # Evitar división por cero
        rms = np.sqrt(np.mean(y**2))
        if rms < 1e-7:
            return y
            
        current_dbfs = 20 * np.log10(rms)
        gain_db = target_dbfs - current_dbfs
        gain_linear = 10 ** (gain_db / 20.0)
        
        y_norm = y * gain_linear
        
        # Limitador suave en picos (Peak ceiling = -0.5 dBFS -> 0.944)
        peak = np.max(np.abs(y_norm))
        if peak > 0.944:
            y_norm = y_norm * (0.944 / peak)
            
        return y_norm

    @classmethod
    def enhance_audio(
        cls,
        audio_input: Union[str, np.ndarray],
        sr: Optional[int] = None,
        config: Optional[AudioEnhancementConfig] = None,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int, str]:
        """
        Ejecuta el pipeline completo de acondicionamiento acústico:
        1. Carga y conversión a mono a 24 kHz.
        2. Filtro paso-banda (75 Hz - 11.5 kHz).
        3. Reducción adaptativa de ruido espectral.
        4. Recorte de silencios extremos (VAD).
        5. Normalización de volumen RMS y control de picos.
        6. Persistencia opcional en archivo WAV limpio.
        
        Retorna (y_limpio, sample_rate, ruta_archivo_limpio).
        """
        if config is None:
            config = AudioEnhancementConfig()
            
        # 1. Cargar señal
        if isinstance(audio_input, str):
            if not os.path.isfile(audio_input):
                raise FileNotFoundError(f"No existe el archivo de audio: {audio_input}")
            # Cargar y remuestrear a target_sr en un solo paso con librosa/soundfile
            y, file_sr = librosa.load(audio_input, sr=config.target_sr, mono=True)
            sr = config.target_sr
        elif isinstance(audio_input, np.ndarray):
            if sr is None:
                raise ValueError("Debe especificarse 'sr' cuando 'audio_input' es un array NumPy.")
            y = audio_input.astype(np.float32)
            if y.ndim > 1:
                y = np.mean(y, axis=-1)  # Convertir a mono
            if sr != config.target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=config.target_sr)
                sr = config.target_sr
        else:
            raise TypeError("audio_input debe ser una ruta de archivo (str) o un array NumPy.")

        # Si el audio está prácticamente mudo, devolverlo sin procesar
        if np.max(np.abs(y)) < 1e-6:
            clean_path = output_path or tempfile.mktemp(suffix="_clean.wav")
            sf.write(clean_path, y, sr)
            return y, sr, clean_path

        # 2. Filtro paso-banda (Aislamiento de rango vocal)
        y = cls._apply_bandpass_filter(y, sr, lowcut=config.high_pass_hz, highcut=config.low_pass_hz)

        # 3. Reducción de ruido espectral adaptativa
        if config.noise_reduction_strength > 0:
            y = cls._spectral_denoise(y, sr, strength=config.noise_reduction_strength)

        # 4. Recorte de silencios
        if config.trim_silence:
            y = cls._trim_silences(y, sr, top_db=config.trim_top_db)

        # 5. Normalización de sonoridad
        y = cls._normalize_loudness(y, target_dbfs=config.target_dbfs)

        # 6. Guardar archivo limpio
        if output_path is None:
            temp_dir = os.path.join(tempfile.gettempdir(), "xtts_enhanced_refs")
            os.makedirs(temp_dir, exist_ok=True)
            clean_path = os.path.join(temp_dir, f"clean_{os.getpid()}_{np.random.randint(1000, 9999)}.wav")
        else:
            clean_path = output_path
            os.makedirs(os.path.dirname(os.path.abspath(clean_path)), exist_ok=True)
            
        sf.write(clean_path, y, sr, subtype="PCM_16")
        return y, sr, clean_path


if __name__ == "__main__":
    print("Módulo audio_enhancer cargado correctamente.")
