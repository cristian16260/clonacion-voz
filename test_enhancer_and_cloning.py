#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_enhancer_and_cloning.py
----------------------------
Suite de validación automatizada para el pipeline de optimización acústica
y extracción de perfiles de alta fidelidad en XTTS-v2.
"""

import os
import sys
import time
import tempfile
import numpy as np
import soundfile as sf
import torch

# Forzar UTF-8 en Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from audio_enhancer import AudioEnhancer, AudioEnhancementConfig
from xtts_engine import XTTSEngine, VOICES_DIR, OUTPUTS_DIR


def test_1_synthetic_signal_denoising_and_filtering():
    """Valida los filtros pasa-banda, denoising STFT y normalización en señal sintética ruidosa."""
    print("\n--- Test 1: Filtrado, Denoising y Normalización en Señal de Prueba ---")
    sr = 24000
    duration_s = 3.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    
    # Señal: Vocal fundamental (220 Hz + 440 Hz) + Ruido eléctrico (50 Hz) + Siseo agudo (14 kHz) + Ruido blanco
    voice_component = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    hum_50hz = 0.3 * np.sin(2 * np.pi * 50 * t)
    hiss_14khz = 0.2 * np.sin(2 * np.pi * 14000 * t)
    noise_white = 0.08 * np.random.normal(0, 1, len(t))
    noisy_signal = voice_component + hum_50hz + hiss_14khz + noise_white
    
    # Guardar en archivo temporal
    temp_in = os.path.join(tempfile.gettempdir(), "test_noisy_input.wav")
    sf.write(temp_in, noisy_signal, sr)
    
    t0 = time.time()
    clean_y, clean_sr, clean_path = AudioEnhancer.enhance_audio(temp_in)
    dt = time.time() - t0
    
    assert clean_sr == 24000, f"Error: Sample rate esperado 24000, obtenido {clean_sr}"
    assert os.path.isfile(clean_path), f"Error: No se generó el archivo limpio en {clean_path}"
    assert np.max(np.abs(clean_y)) <= 0.95, f"Error: Posible clipping detectado ({np.max(np.abs(clean_y))})"
    assert not np.isnan(clean_y).any(), "Error: Señal contiene valores NaN"
    assert dt < 1.5, f"Error: Tiempo de procesamiento excedió el umbral ({dt:.3f}s > 1.5s)"
    
    print(f"✓ Test 1 Pasado: Señal sintética acondicionada en {dt:.3f}s (SR={clean_sr}, Peak={np.max(np.abs(clean_y)):.2f})")
    try:
        os.remove(temp_in)
        os.remove(clean_path)
    except Exception:
        pass


def test_2_real_reference_audio_enhancement():
    """Valida el acondicionamiento acústico sobre audio real de locutor (reference_audios/Jair.wav)."""
    print("\n--- Test 2: Acondicionamiento de Audio Real de Referencia ---")
    real_audio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_audios", "Jair.wav")
    assert os.path.isfile(real_audio), f"No existe el archivo de audio de prueba: {real_audio}"
    
    t0 = time.time()
    clean_y, clean_sr, clean_path = AudioEnhancer.enhance_audio(real_audio)
    dt = time.time() - t0
    
    orig_info = sf.info(real_audio)
    clean_info = sf.info(clean_path)
    
    assert clean_sr == 24000, f"Error en SR: {clean_sr}"
    assert clean_info.duration > 5.0, f"Error: Duración limpia muy corta ({clean_info.duration:.2f}s)"
    assert dt < 3.0, f"Error: Tiempo de procesamiento ({dt:.3f}s) excede NF1 (< 3.0s)"
    
    print(f"✓ Test 2 Pasado: Audio real ({orig_info.duration:.2f}s) procesado a ({clean_info.duration:.2f}s) en {dt:.3f}s.")
    try:
        os.remove(clean_path)
    except Exception:
        pass


def test_3_voice_profile_extraction_with_enhancement():
    """Valida la extracción de un perfil .pt con metadata extendida (enhanced=True)."""
    print("\n--- Test 3: Extracción de Perfil de Voz .pt con Limpieza IA ---")
    real_audio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_audios", "Jair.wav")
    engine = XTTSEngine.get_instance()
    
    voice_name = "test_locutor_enhanced"
    pt_path, dur = engine.extract_voice_profile(
        audio_paths=real_audio,
        voice_name=voice_name,
        temperature=0.60,
        speed=1.0,
        enhance_audio=True
    )
    
    assert os.path.isfile(pt_path), f"Error: No se creó el archivo {pt_path}"
    
    file_size_kb = os.path.getsize(pt_path) / 1024
    assert file_size_kb < 150.0, f"Error: El tamaño del perfil ({file_size_kb:.2f} KB) excede NF2 (< 150 KB)"
    
    # Cargar y verificar estructura interna
    data = torch.load(pt_path, weights_only=False)
    assert "gpt_cond_latent" in data, "Falta gpt_cond_latent en el archivo .pt"
    assert "speaker_embedding" in data, "Falta speaker_embedding en el archivo .pt"
    assert "metadata" in data, "Falta metadata en el archivo .pt"
    assert data["metadata"].get("enhanced") is True, "Metadata no registra 'enhanced: True'"
    assert data["metadata"].get("calibrated_temperature") == 0.60, "Temperatura errónea en metadata"
    
    print(f"✓ Test 3 Pasado: Perfil '{voice_name}.pt' generado ({file_size_kb:.2f} KB), enhanced=True, dur={dur:.2f}s.")


def test_4_tts_synthesis_with_enhanced_profile():
    """Valida que el perfil generado con audio limpio sintetice texto en español con alta velocidad (RTF < 0.6)."""
    print("\n--- Test 4: Inferencia y Síntesis Directa con Perfil Acondicionado ---")
    engine = XTTSEngine.get_instance()
    voice_name = "test_locutor_enhanced"
    
    test_text = "El sistema de clonación de voz con optimización acústica inteligente ha sido validado correctamente."
    
    t0 = time.time()
    res = engine.synthesize(
        text=test_text,
        voice_identifier=voice_name,
        language="es"
    )
    total_time = time.time() - t0
    
    assert os.path.isfile(res["output_path"]), f"No se generó el audio de salida en: {res['output_path']}"
    assert res["duration_s"] > 2.0, f"Duración muy corta: {res['duration_s']:.2f}s"
    assert res["rtf"] < 0.60, f"Error: RTF ({res['rtf']:.3f}) superó el umbral de 0.60"
    
    print(f"✓ Test 4 Pasado: Audio sintetizado ({res['duration_s']:.2f}s) en {res['infer_time_s']:.2f}s (RTF: {res['rtf']:.3f}).")


if __name__ == "__main__":
    print("==========================================================")
    print("EJECUTANDO SUITE DE VALIDACIÓN: OPTIMIZACIÓN ACÚSTICA XTTS")
    print("==========================================================")
    
    test_1_synthetic_signal_denoising_and_filtering()
    test_2_real_reference_audio_enhancement()
    test_3_voice_profile_extraction_with_enhancement()
    test_4_tts_synthesis_with_enhanced_profile()
    
    print("\n==========================================================")
    print("TODOS LOS TESTS (4/4) COMPLETADOS CON ÉXITO [ESTADO: PASÓ]")
    print("==========================================================")
