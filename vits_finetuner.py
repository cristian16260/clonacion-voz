#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vits_finetuner.py
-----------------
Motor de Micro Fine-Tuning para la arquitectura VITS en español:
1. Preprocesa los audios del usuario con AudioEnhancer y transcribe con Whisper (DatasetPreparer).
2. Carga el checkpoint base oficial en español ('tts_models/es/css10/vits').
3. Ejecuta el ajuste fino (transfer learning) de la red VITS en GPU (CUDA) o CPU.
4. Exporta el modelo optimizado (< 150 MB) a 'trained_models/vits/<nombre_voz>/' con metadatos.
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
from torch.utils.data import DataLoader

# Forzar codificación UTF-8 en Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from TTS.tts.configs.vits_config import VitsConfig
import TTS.tts.datasets.dataset as tts_dataset_mod
import TTS.tts.models.vits as vits_mod
from TTS.tts.models.vits import Vits
from TTS.utils.manage import ModelManager

from audio_enhancer import AudioEnhancer
from dataset_preparer import DatasetPreparer

# Garantizar carga y cálculo de frames de audio con soundfile (sin depender de torchcodec)
def _safe_get_audio_size(audiopath) -> int:
    try:
        return int(sf.info(str(audiopath)).frames)
    except Exception:
        return 22050 * 5

def _safe_audio_load(file_path, *args, **kwargs):
    wav, sr = sf.read(file_path, dtype="float32")
    if len(wav.shape) == 1:
        tensor = torch.from_numpy(wav).unsqueeze(0)
    else:
        tensor = torch.from_numpy(wav.T)
    return tensor, sr

tts_dataset_mod.get_audio_size = _safe_get_audio_size
if hasattr(vits_mod, "load_audio"):
    vits_mod.load_audio = _safe_audio_load

try:
    import torchaudio
    torchaudio.load = _safe_audio_load
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_VITS_DIR = os.path.join(BASE_DIR, "trained_models", "vits")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
BASE_MODEL_NAME = "tts_models/es/css10/vits"

os.makedirs(TRAINED_VITS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)


class VITSFineTuner:
    """Ajustador fino (Fine-Tuning) para modelos VITS en español."""

    @classmethod
    def parse_metadata_csv(cls, metadata_csv_path: str) -> List[Dict]:
        """Parsea el archivo metadata.csv generado por DatasetPreparer."""
        if not os.path.isfile(metadata_csv_path):
            raise FileNotFoundError(f"No existe el archivo metadata.csv en: {metadata_csv_path}")

        dataset_root = os.path.dirname(metadata_csv_path)
        samples = []
        with open(metadata_csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                parts = line_str.split("|")
                if len(parts) >= 2:
                    raw_path = parts[0].strip()
                    text = parts[1].strip()
                    speaker = parts[2].strip() if len(parts) > 2 else "speaker"

                    if raw_path.lower() in ("audio_file", "audio", "file", "filename"):
                        continue

                    # Resolver ruta de audio buscando en varios posibles orígenes
                    resolved_path = None
                    candidates = [
                        raw_path,
                        os.path.join(BASE_DIR, raw_path),
                        os.path.join(dataset_root, raw_path),
                        os.path.join(dataset_root, "wavs", os.path.basename(raw_path)),
                        os.path.join(dataset_root, os.path.basename(raw_path))
                    ]
                    for cand in candidates:
                        if os.path.isfile(cand):
                            resolved_path = os.path.abspath(cand)
                            break

                    if resolved_path and text:
                        base_name = os.path.splitext(os.path.basename(resolved_path))[0]
                        samples.append({
                            "audio_file": resolved_path,
                            "text": text,
                            "speaker_name": speaker,
                            "language": "es",
                            "audio_unique_name": base_name,
                            "root_path": dataset_root
                        })
        return samples

    @classmethod
    def train_voice(
        cls,
        voice_name: str,
        dataset_dir: str,
        epochs: int = 8,
        batch_size: int = 2,
        lr_gen: float = 5e-5,
        lr_disc: float = 5e-5,
        progress_callback=None
    ) -> Dict[str, Union[str, float, int]]:
        """
        Ejecuta el ciclo de Fine-Tuning de VITS sobre el modelo base en español.
        
        Parámetros:
            voice_name: Identificador único de la voz.
            dataset_dir: Carpeta con metadata.csv y wavs/ del locutor.
            epochs: Número de épocas de ajuste (8-12 recomendado).
            batch_size: Tamaño de batch para entrenamiento.
            lr_gen: Learning rate del generador VITS.
            lr_disc: Learning rate del discriminador VITS.
            progress_callback: Función callback(progress_fraction, status_text).
            
        Retorna:
            Diccionario con información del modelo entrenado y ruta guardada.
        """
        t_start = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clean_voice_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in voice_name.strip())
        output_model_dir = os.path.join(TRAINED_VITS_DIR, clean_voice_name)
        os.makedirs(output_model_dir, exist_ok=True)

        if progress_callback:
            progress_callback(0.05, "Descargando / Localizando modelo base VITS en español...")

        # 1. Localizar modelo base y config oficial
        manager = ModelManager()
        base_model_path, base_config_path, _ = manager.download_model(BASE_MODEL_NAME)

        # 2. Parsear muestras del dataset
        metadata_csv = os.path.join(dataset_dir, "metadata.csv")
        samples = cls.parse_metadata_csv(metadata_csv)
        if not samples:
            raise ValueError(f"No se encontraron muestras de audio y texto válidas en '{metadata_csv}'.")

        if progress_callback:
            progress_callback(0.12, f"Cargando configuración y pesos base ({len(samples)} clips de audio)...")

        # 3. Configurar VitsConfig para entrenamiento
        config = VitsConfig()
        config.load_json(base_config_path)
        config.model_args.init_discriminator = True
        config.batch_size = int(batch_size)
        config.eval_batch_size = int(batch_size)
        config.run_eval = False
        config.num_loader_workers = 0
        config.num_eval_loader_workers = 0
        config.phoneme_cache_path = None
        config.start_by_longest = False
        config.min_audio_len = 500
        config.max_audio_len = 22050 * 30
        config.min_text_len = 1
        config.max_text_len = 1000
        config.lr_gen = lr_gen
        config.lr_disc = lr_disc

        # 4. Inicializar modelo VITS y cargar pesos base
        model = Vits.init_from_config(config)
        model.load_checkpoint(config, checkpoint_path=base_model_path, eval=False, strict=False)
        model.init_for_training()

        if device == "cuda":
            model.cuda()

        # 5. Configurar speaker_manager para reconocer la voz del dataset
        if model.speaker_manager is not None:
            model.speaker_manager.name_to_id[clean_voice_name] = 0
            if hasattr(model.speaker_manager, "speaker_names") and clean_voice_name not in model.speaker_manager.speaker_names:
                model.speaker_manager.speaker_names.append(clean_voice_name)

        # 6. Obtener Data Loader y Criterios
        criterion = model.get_criterion()
        optimizer_disc, optimizer_gen = model.get_optimizer()

        data_loader = model.get_data_loader(
            config=config,
            assets={},
            is_eval=False,
            samples=samples,
            verbose=False,
            num_gpus=1
        )

        num_batches = len(data_loader)
        total_steps = epochs * num_batches
        step_counter = 0
        last_loss = 0.0

        print(f"[VITSFineTuner] Iniciando Fine-Tuning de '{clean_voice_name}' en {device.upper()}: {epochs} épocas, {len(samples)} clips.")

        # 7. Bucle de Entrenamiento
        model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss_g = 0.0
            epoch_loss_d = 0.0
            
            for batch_idx, batch in enumerate(data_loader):
                step_counter += 1
                
                # Formatear batch (speaker/language/d_vectors) y mover al dispositivo
                batch = model.format_batch(batch)
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                batch = model.format_batch_on_device(batch)
                
                # Paso 0: Discriminador VITS (optimizer_idx = 0)
                optimizer_disc.zero_grad()
                _, loss_dict_d = model.train_step(batch, criterion, optimizer_idx=0)
                loss_d = loss_dict_d.get("loss", list(loss_dict_d.values())[0])
                loss_d.backward()
                optimizer_disc.step()
                epoch_loss_d += loss_d.item()

                # Paso 1: Generador VITS (optimizer_idx = 1)
                optimizer_gen.zero_grad()
                _, loss_dict_g = model.train_step(batch, criterion, optimizer_idx=1)
                loss_g = loss_dict_g.get("loss", list(loss_dict_g.values())[0])
                loss_g.backward()
                optimizer_gen.step()
                epoch_loss_g += loss_g.item()
                last_loss = loss_g.item()

                if progress_callback:
                    prog_frac = 0.15 + (0.75 * (step_counter / float(total_steps)))
                    elapsed = time.time() - t_start
                    eta = (elapsed / step_counter) * (total_steps - step_counter) if step_counter > 0 else 0
                    progress_callback(
                        min(0.92, prog_frac),
                        f"Época {epoch}/{epochs} | Batch {batch_idx+1}/{num_batches} | Loss Gen: {loss_g.item():.3f} | ETA: {int(eta)}s"
                    )

            avg_loss_g = epoch_loss_g / max(1, num_batches)
            avg_loss_d = epoch_loss_d / max(1, num_batches)
            print(f"[VITSFineTuner] Época {epoch:02d}/{epochs:02d} - Loss Gen: {avg_loss_g:.4f} | Loss Disc: {avg_loss_d:.4f}")

        if progress_callback:
            progress_callback(0.94, "Guardando checkpoint optimizado y configuración...")

        # 8. Guardar modelo afinado optimizado para inferencia (<110 MB)
        out_model_pth = os.path.join(output_model_dir, "model.pth")
        out_config_json = os.path.join(output_model_dir, "config.json")
        out_meta_json = os.path.join(output_model_dir, "metadata.json")

        # Filtrar pesos del discriminador para conservar solo el generador para inferencia
        inference_state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("disc.")}
        torch.save({"model": inference_state_dict}, out_model_pth)
        
        # Guardar config con init_discriminator=False para despliegue ligero
        config.model_args.init_discriminator = False
        config.save_json(out_config_json)

        # Guardar metadatos descriptivos
        dur_total = time.time() - t_start
        size_mb = os.path.getsize(out_model_pth) / (1024 * 1024)
        metadata = {
            "voice_name": clean_voice_name,
            "base_model": BASE_MODEL_NAME,
            "epochs": epochs,
            "final_loss": round(last_loss, 4),
            "num_training_clips": len(samples),
            "training_duration_s": round(dur_total, 2),
            "model_size_mb": round(size_mb, 2),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device.upper()
        }
        with open(out_meta_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[VITSFineTuner] Fine-tuning completado con éxito. Guardado en '{output_model_dir}' ({size_mb:.2f} MB).")
        
        if progress_callback:
            progress_callback(1.0, f"¡Entrenamiento completado! Modelo '{clean_voice_name}' listo ({size_mb:.1f} MB).")

        return {
            "voice_name": clean_voice_name,
            "output_dir": output_model_dir,
            "model_path": out_model_pth,
            "config_path": out_config_json,
            "size_mb": round(size_mb, 2),
            "duration_s": round(dur_total, 2),
            "metadata": metadata
        }

    @classmethod
    def train_from_raw_audios(
        cls,
        voice_name: str,
        audio_paths: Union[str, List[str]],
        epochs: int = 8,
        batch_size: int = 2,
        progress_callback=None
    ) -> Dict[str, Union[str, float, int]]:
        """
        Pipeline integral de Fine-Tuning:
        1. Limpieza y preparación de dataset a partir de los audios en bruto.
        2. Ejecución del Fine-Tuning con VITS.
        """
        clean_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in voice_name.strip())
        
        if progress_callback:
            progress_callback(0.02, "Iniciando acondicionamiento acústico y segmentación...")

        # 1. Generar dataset usando DatasetPreparer
        ds_info = DatasetPreparer.prepare_dataset(
            audio_paths=audio_paths,
            voice_name=clean_name,
            progress_callback=progress_callback
        )

        dataset_dir = ds_info["dataset_dir"]

        # 2. Entrenar red VITS
        result = cls.train_voice(
            voice_name=clean_name,
            dataset_dir=dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            progress_callback=progress_callback
        )
        return result


if __name__ == "__main__":
    print("=== Test VITSFineTuner ===")
    print("Módulo VITSFineTuner importado y listo.")
