#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_finetuning_pipeline.py
---------------------------
Suite de validación automatizada para el sistema de Micro Fine-Tuning de XTTS-v2:
1. Verificación de preparación de dataset acústico y ASR Whisper local.
2. Verificación del entrenamiento de GPT en GPU RTX 5070 con optimizador AdamW.
3. Verificación de persistencia de artefactos en 'trained_models/<nombre_voz>/'.
4. Verificación de carga dinámica y síntesis de audio de alta fidelidad.
"""

import os
import sys
import unittest
import numpy as np
import soundfile as sf
import torch

# Forzar UTF-8 en consola de Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dataset_preparer import DatasetPreparer, DATASETS_DIR
from xtts_finetuner import XTTSFineTuner, TRAINED_MODELS_DIR
from xtts_engine import XTTSEngine, VOICES_DIR, OUTPUTS_DIR


class TestFineTuningPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepara las carpetas y archivos temporales para las pruebas."""
        cls.test_voice_name = "Test_Speaker_Pipeline"
        cls.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_temp_audio")
        os.makedirs(cls.temp_dir, exist_ok=True)
        
        # Generar un archivo de audio sintético para testing (onda de voz con armónicos)
        cls.sample_wav_path = os.path.join(cls.temp_dir, "sample_voice.wav")
        sr = 22050
        duration = 6.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Señal vocal sintética con formantes armónicos
        audio = 0.4 * np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 600 * t)
        # Añadir pausas para testear segmentación VAD
        audio[int(sr * 2.5):int(sr * 3.0)] = 0.0
        sf.write(cls.sample_wav_path, audio.astype(np.float32), sr)

    def test_01_dataset_preparer(self):
        """Verifica que DatasetPreparer limpie, segmente y transcriba audios."""
        res = DatasetPreparer.prepare_dataset(
            audio_paths=[self.sample_wav_path],
            voice_name=self.test_voice_name
        )
        
        self.assertIn("dataset_dir", res)
        self.assertIn("metadata_csv", res)
        self.assertTrue(os.path.isdir(res["dataset_dir"]))
        self.assertTrue(os.path.isfile(res["metadata_csv"]))
        self.assertGreaterEqual(res["total_clips"], 1)
        
        # Verificar contenido de metadata.csv
        with open(res["metadata_csv"], "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 1, "metadata.csv debe tener al menos la cabecera y 1 fila")
        print(f"[TEST 1 PASS] Dataset preparado con {res['total_clips']} clips.")

    def test_02_gpu_micro_finetuner(self):
        """Verifica que el FineTuner ejecute épocas en GPU y guarde artefactos."""
        dataset_dir = os.path.join(DATASETS_DIR, self.test_voice_name)
        res = XTTSFineTuner.train_voice(
            voice_name=self.test_voice_name,
            dataset_dir=dataset_dir,
            epochs=2,
            batch_size=1,
            lr=5e-6
        )
        
        self.assertEqual(res["voice_name"], self.test_voice_name)
        self.assertTrue(os.path.isfile(res["model_path"]))
        self.assertTrue(os.path.isfile(res["speaker_path"]))
        self.assertFalse(np.isnan(res["final_loss"]), "El Loss no debe ser NaN")
        self.assertGreater(res["final_loss"], 0.0)
        
        # Verificar que existan todos los artefactos en trained_models/
        model_dir = res["model_dir"]
        self.assertTrue(os.path.isfile(os.path.join(model_dir, "model.pth")))
        self.assertTrue(os.path.isfile(os.path.join(model_dir, "speaker_embedding.pt")))
        self.assertTrue(os.path.isfile(os.path.join(model_dir, "vocab.json")))
        self.assertTrue(os.path.isfile(os.path.join(model_dir, "config.json")))
        self.assertTrue(os.path.isfile(os.path.join(model_dir, "metadata.json")))
        print(f"[TEST 2 PASS] Fine-tuning completado en GPU. Loss final: {res['final_loss']:.4f}")

    def test_03_engine_trained_model_synthesis(self):
        """Verifica que XTTSEngine reconozca, cargue y sintetice con el modelo entrenado."""
        engine = XTTSEngine.get_instance()
        available_models = engine.get_available_trained_models()
        self.assertIn(self.test_voice_name, available_models)
        
        synth_text = "Esta es una prueba de sintesis con la voz afinada en local."
        res = engine.synthesize(
            text=synth_text,
            voice_identifier=self.test_voice_name,
            language="es"
        )
        
        self.assertIn("output_path", res)
        self.assertTrue(os.path.isfile(res["output_path"]))
        self.assertGreater(res["duration_s"], 1.0)
        self.assertLess(res["rtf"], 0.50, "El RTF en RTX 5070 debe ser menor a 0.50")
        self.assertTrue(res.get("metadata", {}).get("is_finetuned", False))
        print(f"[TEST 3 PASS] Síntesis exitosa con modelo afinado. RTF: {res['rtf']:.3f}")

    def test_04_fallback_and_restoration(self):
        """Verifica que el motor pueda conmutar entre modelos afinados y perfiles .pt sin corrupción."""
        engine = XTTSEngine.get_instance()
        # Cargar modelo entrenado
        engine.load_voice_profile(self.test_voice_name)
        self.assertEqual(engine._current_finetuned_model, self.test_voice_name)
        
        # Restaurar modelo base
        engine.restore_base_model()
        self.assertIsNone(engine._current_finetuned_model)
        print("[TEST 4 PASS] Conmutación bidireccional y restauración de modelo base verificada.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
