#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_vits_es.py
---------------
Suite de pruebas automatizadas para el motor VITS en español:
1. Verificación de aislamiento e independencia arquitectónica (sin dependencias de XTTS).
2. Verificación de tamaño de modelo en disco (< 160 MB).
3. Verificación de síntesis, frecuencia de muestreo (22050 Hz) y métricas de RTF.
4. Verificación del pipeline de Fine-Tuning y carga de modelos afinados.
"""

import json
import os
import sys
import time
import unittest
import numpy as np
import soundfile as sf
import torch

# Forzar codificación UTF-8 en Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class TestVITSModule(unittest.TestCase):
    """Pruebas unitarias y de integración para VITS en Español."""

    def test_01_isolation(self):
        """Verifica que el motor VITS está desacoplado y no importa XTTS."""
        import vits_engine
        import vits_finetuner
        
        # Comprobar que xtts_engine no fue importado accidentalmente en sys.modules
        self.assertNotIn("xtts_engine", sys.modules, "vits_engine o vits_finetuner importaron xtts_engine indebidamente.")
        print("✅ Test 1: Aislamiento arquitectónico verificado (Cero acoplamiento con XTTS).")

    def test_02_model_size(self):
        """Verifica que el tamaño del modelo base en disco sea < 160 MB."""
        from TTS.utils.manage import ModelManager
        manager = ModelManager()
        model_path, config_path, _ = manager.download_model("tts_models/es/css10/vits")
        
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📦 Tamaño del modelo base VITS: {size_mb:.2f} MB")
        self.assertLess(size_mb, 160.0, f"El modelo excede el límite de 160 MB ({size_mb} MB)")
        print("✅ Test 2: Criterio de peso ligero superado (< 160 MB).")

    def test_03_synthesis_and_metrics(self):
        """Verifica la síntesis con el modelo base, SR 22050 Hz y métricas de RTF."""
        from vits_engine import VITSEngine
        engine = VITSEngine.get_instance()
        
        test_text = "Esta es una verificación automatizada del motor VITS en idioma español."
        audio_array, sr, metrics = engine.synthesize(
            text=test_text,
            model_name="base_es",
            speed=1.0,
            output_filename="test_vits_synth.wav"
        )
        
        self.assertEqual(sr, 22050, f"Sample rate esperado: 22050, obtenido: {sr}")
        self.assertIsInstance(audio_array, np.ndarray)
        self.assertGreater(len(audio_array), 0)
        self.assertGreater(metrics["audio_duration_s"], 1.0)
        self.assertTrue(os.path.isfile(metrics["output_path"]))
        
        print(f"⏱️ Inferencia: {metrics['inference_time_s']}s | Duración: {metrics['audio_duration_s']}s | RTF: {metrics['rtf']} | Dispositivo: {metrics['device']}")
        print("✅ Test 3: Síntesis de voz y cálculo de métricas validado exitosamente.")

    def test_04_finetuning_pipeline_and_load(self):
        """Verifica el ciclo de fine-tuning y posterior carga en VITSEngine."""
        from vits_finetuner import VITSFineTuner
        from vits_engine import VITSEngine
        
        ref_audio = os.path.join(BASE_DIR, "reference_audios", "Jair.wav")
        if not os.path.isfile(ref_audio):
            # Crear un audio dummy si no existe Jair.wav
            ref_audio = os.path.join(BASE_DIR, "outputs", "vits", "test_vits_synth.wav")
            
        test_voice = "test_vits_voice"
        
        # Ejecutar 1 época de fine-tuning para validación rápida
        print("🚀 Ejecutando prueba de Fine-Tuning (1 época)...")
        result = VITSFineTuner.train_from_raw_audios(
            voice_name=test_voice,
            audio_paths=[ref_audio],
            epochs=1,
            batch_size=1
        )
        
        self.assertTrue(os.path.isfile(result["model_path"]))
        self.assertTrue(os.path.isfile(result["config_path"]))
        self.assertLess(result["size_mb"], 160.0)
        
        # Probar carga e inferencia con el nuevo modelo afinado
        engine = VITSEngine.get_instance()
        all_models = engine.get_available_trained_models()
        self.assertIn(test_voice, all_models)
        
        wav, sr, met = engine.synthesize(
            text="Prueba con el nuevo modelo afinado en español.",
            model_name=test_voice,
            output_filename="test_vits_finetuned_synth.wav"
        )
        self.assertGreater(len(wav), 0)
        print("✅ Test 4: Pipeline de Fine-Tuning y carga de modelo afinado completado con éxito.")


if __name__ == "__main__":
    unittest.main()
