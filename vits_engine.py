#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vits_engine.py
--------------
Motor central de síntesis e inferencia de voz en español basado en la arquitectura VITS.
- Ultraligero (~103 MB de checkpoint base) y optimizado para CPU y GPU (CUDA).
- Inferencia rápida con control de velocidad (speed) y variabilidad acústica (noise_scale).
- Gestión de modelos afinados (fine-tuned) en trained_models/vits/.
- Desacoplado de XTTS para despliegue limpio y microservicios independientes (ej: Amazon SageMaker).
"""

import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

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

from TTS.api import TTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "vits")
TRAINED_VITS_DIR = os.path.join(BASE_DIR, "trained_models", "vits")
BASE_MODEL_NAME = "tts_models/es/css10/vits"

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(TRAINED_VITS_DIR, exist_ok=True)


class VITSEngine:
    """Motor de síntesis e inferencia para VITS en español."""

    _instance: Optional["VITSEngine"] = None

    def __init__(self, base_model_name: str = BASE_MODEL_NAME):
        self.base_model_name = base_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._current_loaded_model_id: Optional[str] = None
        self.tts: Optional[TTS] = None
        self.sample_rate: int = 22050
        
        print(f"[VITSEngine] Inicializando motor VITS en dispositivo: {self.device.upper()}")
        self._load_base_model()

    @classmethod
    def get_instance(cls, base_model_name: str = BASE_MODEL_NAME) -> "VITSEngine":
        """Devuelve la instancia Singleton del motor VITS."""
        if cls._instance is None:
            cls._instance = cls(base_model_name=base_model_name)
        return cls._instance

    def _load_base_model(self) -> None:
        """Carga en memoria el modelo base oficial de VITS en español."""
        try:
            print(f"[VITSEngine] Cargando modelo base '{self.base_model_name}'...")
            self.tts = TTS(
                model_name=self.base_model_name,
                progress_bar=False,
                gpu=(self.device == "cuda")
            )
            if hasattr(self.tts, "synthesizer") and hasattr(self.tts.synthesizer, "output_sample_rate"):
                self.sample_rate = int(self.tts.synthesizer.output_sample_rate)
            self._current_loaded_model_id = "base_es"
            print(f"[VITSEngine] Modelo base '{self.base_model_name}' cargado exitosamente. (SR: {self.sample_rate} Hz)")
        except Exception as e:
            print(f"[VITSEngine] Error al cargar el modelo base VITS: {e}")
            raise

    def load_model(self, model_identifier: str = "base_es") -> bool:
        """
        Carga un modelo específico (base_es o nombre de modelo afinado en trained_models/vits/).
        """
        if model_identifier == self._current_loaded_model_id and self.tts is not None:
            return True

        if model_identifier in ("base_es", "es_css10", BASE_MODEL_NAME):
            self._load_base_model()
            return True

        # Buscar en trained_models/vits/
        clean_name = os.path.splitext(os.path.basename(model_identifier))[0]
        model_dir = os.path.join(TRAINED_VITS_DIR, clean_name)
        
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"No se encontró el modelo afinado VITS en '{model_dir}'.")

        model_pth = os.path.join(model_dir, "model.pth")
        config_json = os.path.join(model_dir, "config.json")
        
        if not os.path.isfile(model_pth) or not os.path.isfile(config_json):
            # Intentar buscar checkpoints alternativos (.pth o best_model.pth)
            found_pth = None
            for fname in os.listdir(model_dir):
                if fname.endswith(".pth") or fname.endswith(".pth.tar"):
                    found_pth = os.path.join(model_dir, fname)
                    break
            if found_pth and os.path.isfile(config_json):
                model_pth = found_pth
            else:
                raise FileNotFoundError(f"Faltan 'model.pth' o 'config.json' en '{model_dir}'.")

        print(f"[VITSEngine] Cargando modelo afinado VITS '{clean_name}' desde: {model_dir}")
        self.tts = TTS(
            model_path=model_pth,
            config_path=config_json,
            progress_bar=False,
            gpu=(self.device == "cuda")
        )
        if hasattr(self.tts, "synthesizer") and hasattr(self.tts.synthesizer, "output_sample_rate"):
            self.sample_rate = int(self.tts.synthesizer.output_sample_rate)
            
        self._current_loaded_model_id = clean_name
        print(f"[VITSEngine] Modelo afinado '{clean_name}' cargado exitosamente.")
        return True

    @staticmethod
    def get_available_trained_models() -> List[str]:
        """Devuelve la lista de nombres de modelos VITS entrenados disponibles."""
        if not os.path.isdir(TRAINED_VITS_DIR):
            return []
        
        models = []
        for d in os.listdir(TRAINED_VITS_DIR):
            sub = os.path.join(TRAINED_VITS_DIR, d)
            if os.path.isdir(sub):
                has_model = any(f.endswith((".pth", ".pth.tar")) for f in os.listdir(sub))
                has_config = os.path.isfile(os.path.join(sub, "config.json"))
                if has_model and has_config:
                    models.append(d)
        return sorted(models)

    @staticmethod
    def get_all_models_list() -> List[str]:
        """Devuelve la lista formateada para interfaz de todos los modelos VITS disponibles."""
        items = ["🇪🇸 [Base Oficial] es/css10 (Voz Femenina Estándar)"]
        for tm in VITSEngine.get_available_trained_models():
            items.append(f"⭐ [Voz Entrenada VITS] {tm}")
        return items

    @staticmethod
    def extract_model_id(formatted_str: str) -> str:
        """Extrae el identificador limpio a partir del texto con formato de la UI."""
        if not formatted_str or formatted_str.startswith("("):
            return "base_es"
        if "es/css10" in formatted_str or "[Base Oficial]" in formatted_str:
            return "base_es"
        if "[Voz Entrenada VITS] " in formatted_str:
            return formatted_str.split("[Voz Entrenada VITS] ")[-1].strip()
        return formatted_str.strip()

    def get_model_info(self, model_identifier: str) -> Dict[str, Union[str, float, int]]:
        """Obtiene información y metadatos de un modelo VITS."""
        clean_id = self.extract_model_id(model_identifier)
        if clean_id == "base_es":
            return {
                "name": "es/css10/vits",
                "type": "Base Oficial Coqui VITS",
                "language": "Español (es)",
                "size_mb": 103.5,
                "sample_rate": self.sample_rate,
                "description": "Modelo base oficial entrenado sobre el corpus CSS10 en español."
            }
            
        model_dir = os.path.join(TRAINED_VITS_DIR, clean_id)
        model_pth = os.path.join(model_dir, "model.pth")
        meta_json = os.path.join(model_dir, "metadata.json")
        
        size_mb = 0.0
        if os.path.isfile(model_pth):
            size_mb = os.path.getsize(model_pth) / (1024 * 1024)
            
        meta = {}
        if os.path.isfile(meta_json):
            try:
                with open(meta_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        return {
            "name": clean_id,
            "type": "Voz Finetuned VITS",
            "language": "Español (es)",
            "size_mb": round(size_mb, 2),
            "sample_rate": self.sample_rate,
            "epochs": meta.get("epochs", "N/A"),
            "loss": meta.get("final_loss", "N/A"),
            "trained_at": meta.get("trained_at", "N/A"),
            "description": f"Modelo afinado sobre {clean_id} a {self.sample_rate} Hz."
        }

    def delete_trained_model(self, model_identifier: str) -> bool:
        """Elimina un modelo afinado del disco."""
        clean_id = self.extract_model_id(model_identifier)
        if clean_id == "base_es":
            return False
            
        model_dir = os.path.join(TRAINED_VITS_DIR, clean_id)
        if os.path.isdir(model_dir):
            if self._current_loaded_model_id == clean_id:
                self._load_base_model()
            shutil.rmtree(model_dir, ignore_errors=True)
            return True
        return False

    def synthesize(
        self,
        text: str,
        model_name: str = "base_es",
        speed: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        output_filename: Optional[str] = None
    ) -> Tuple[np.ndarray, int, Dict[str, Union[float, str]]]:
        """
        Sintetiza texto a audio utilizando el modelo VITS seleccionado.
        
        Parámetros:
            text: Texto en español a pronunciar.
            model_name: 'base_es' o nombre del modelo entrenado.
            speed: Factor de velocidad (1.0 = normal, >1 más rápido, <1 más lento).
            noise_scale: Variabilidad acústica del fonema (0.667 estándar).
            noise_scale_w: Variabilidad de la duración de fonemas (0.8 estándar).
            output_filename: Nombre de archivo WAV opcional para guardar en outputs/vits/.
            
        Retorna:
            (audio_array, sample_rate, metrics_dict)
        """
        clean_text = text.strip() if text else ""
        if not clean_text:
            raise ValueError("El texto a sintetizar no puede estar vacío.")

        model_id = self.extract_model_id(model_name)
        self.load_model(model_id)

        # En VITS de Coqui TTS:
        # length_scale = 1.0 / speed (a menor length_scale, más rápido habla)
        length_scale = 1.0 / max(0.4, min(2.5, speed))

        t0 = time.time()
        
        # Inferencia con Coqui TTS
        wav_list = self.tts.tts(
            text=clean_text,
            length_scale=float(length_scale),
            noise_scale=float(noise_scale),
            noise_scale_w=float(noise_scale_w)
        )
        
        t_infer = time.time() - t0
        audio_array = np.array(wav_list, dtype=np.float32)
        
        # Evitar valores NaN / Inf y normalizar pico si excede 1.0
        audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.99, neginf=-0.99)
        max_val = np.max(np.abs(audio_array))
        if max_val > 1.0:
            audio_array = audio_array / max_val

        dur_s = len(audio_array) / float(self.sample_rate)
        rtf = t_infer / dur_s if dur_s > 0 else 0.0

        # Guardar archivo de salida si se requiere o por defecto con timestamp
        if not output_filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"vits_{model_id}_{timestamp}.wav"
            
        out_path = os.path.join(OUTPUTS_DIR, output_filename)
        sf.write(out_path, audio_array, self.sample_rate, subtype="PCM_16")
        file_size_kb = os.path.getsize(out_path) / 1024.0 if os.path.isfile(out_path) else 0.0

        metrics = {
            "inference_time_s": round(t_infer, 3),
            "audio_duration_s": round(dur_s, 2),
            "rtf": round(rtf, 4),
            "sample_rate": self.sample_rate,
            "output_path": out_path,
            "file_size_kb": round(file_size_kb, 1),
            "device": self.device.upper(),
            "model_id": model_id
        }

        return audio_array, self.sample_rate, metrics


if __name__ == "__main__":
    print("=== Probando VITSEngine ===")
    engine = VITSEngine.get_instance()
    test_phrase = "Hola, esta es una prueba de síntesis en español con el modelo VITS ultraligero."
    wav, sr, met = engine.synthesize(test_phrase)
    print("Métricas de síntesis:", json.dumps(met, indent=2))
    print(f"Audio guardado en: {met['output_path']}")
