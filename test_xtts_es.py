#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_xtts_es.py
---------------
Suite de validación automatizada de XTTS-v2 para clonación de voz en español.
Verifica:
1. Carga de modelo en GPU CUDA (NVIDIA RTX 5070)
2. Extracción y persistencia de perfiles de voz (.pt < 150 KB)
3. Síntesis directa en español con RTF < 0.6
4. Generación por streaming de chunks de audio
5. Flujo de registro y síntesis de la Interfaz Web
"""

import io
import os
import sys
import unittest
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xtts_engine import XTTSEngine, VOICES_DIR, OUTPUTS_DIR
from app_xtts import register_and_extract_voice, synthesize_voice_ui


class TestXTTSv2Spanish(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.ref_audio = os.path.join(cls.base_dir, "reference_audios", "Jair.wav")
        cls.engine = XTTSEngine.get_instance()

    def test_01_cuda_and_model_loading(self):
        """Verifica que el modelo esté cargado en GPU CUDA."""
        self.assertTrue(torch.cuda.is_available(), "CUDA debe estar disponible")
        self.assertEqual(self.engine.device, "cuda")
        device_name = torch.cuda.get_device_name(0)
        self.assertIn("RTX", device_name.upper())

    def test_02_extract_voice_profile(self):
        """Verifica la extracción de perfiles latentes (.pt < 150 KB)."""
        self.assertTrue(os.path.isfile(self.ref_audio), f"Debe existir audio de referencia: {self.ref_audio}")
        
        pt_path, dur = self.engine.extract_voice_profile(self.ref_audio, "test_jair")
        self.assertTrue(os.path.isfile(pt_path))
        
        size_kb = os.path.getsize(pt_path) / 1024
        print(f"\n[Test] Tamaño del perfil .pt extraído: {size_kb:.2f} KB (Límite: < 150 KB)")
        self.assertLess(size_kb, 150, "El perfil .pt debe ser menor a 150 KB")
        
        data = torch.load(pt_path, weights_only=False)
        self.assertIn("gpt_cond_latent", data)
        self.assertIn("speaker_embedding", data)
        self.assertIn("metadata", data)

    def test_03_synthesize_spanish(self):
        """Verifica la síntesis directa en español y la métrica RTF < 0.6."""
        out_wav = os.path.join(self.base_dir, "outputs", "test_synth_out.wav")
        res = self.engine.synthesize(
            text="Esta es una prueba automatizada para validar la calidad y velocidad de XTTS en español.",
            voice_identifier="test_jair",
            output_path=out_wav,
            language="es"
        )
        
        self.assertTrue(os.path.isfile(out_wav))
        self.assertEqual(res["sample_rate"], 24000)
        self.assertGreater(res["duration_s"], 2.0)
        print(f"[Test] Tiempo inferencia: {res['infer_time_s']:.2f}s, Duración audio: {res['duration_s']:.2f}s, RTF: {res['rtf']:.3f}")
        self.assertLess(res["rtf"], 0.6, f"RTF ({res['rtf']:.3f}) debe ser menor a 0.6 en GPU RTX 5070")

    def test_04_streaming_chunks(self):
        """Verifica la generación de streaming por bloques de audio."""
        chunks = []
        for chunk in self.engine.synthesize_stream(
            text="Prueba de streaming por fragmentos en tiempo real.",
            voice_identifier="test_jair",
            language="es",
            stream_chunk_size=20
        ):
            chunks.append(chunk)
            
        self.assertGreater(len(chunks), 1, "El streaming debe emitir múltiples chunks de audio")
        total_samples = sum(len(c) for c in chunks)
        self.assertGreater(total_samples, 24000, "El streaming debe generar al menos 1 segundo de muestras")

    def test_05_ui_flow(self):
        """Verifica el flujo completo de funciones de la UI Gradio."""
        msg, dropdown = register_and_extract_voice(self.ref_audio, "jair_ui_test")
        self.assertIn("guardado exitosamente", msg)
        
        out_path, status, t_infer, dur, rtf, size_str = synthesize_voice_ui(
            voice_name="jair_ui_test",
            text="Prueba desde la interfaz de usuario.",
            speed=1.0
        )
        self.assertIsNotNone(out_path)
        self.assertTrue(os.path.isfile(out_path))
        self.assertIn("exitosamente", status)


if __name__ == "__main__":
    unittest.main(verbosity=2)
