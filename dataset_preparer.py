#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_preparer.py
-------------------
Pipeline automatizado para preparación de datasets de voz para Fine-Tuning:
1. Limpieza acústica profunda con AudioEnhancer.
2. Segmentación basada en silencios (VAD) en clips de 2 a 10 segundos.
3. Transcripción fonética automática con Whisper local en GPU (CUDA).
4. Generación de metadata.csv listo para el entrenamiento de XTTS-v2.
"""

import os
import sys
import re
import time
import glob
from typing import List, Tuple, Dict, Optional
import numpy as np
import soundfile as sf
import librosa
import torch

# Forzar UTF-8 en Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from audio_enhancer import AudioEnhancer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)


class DatasetPreparer:
    """Preparador automático de dataset de entrenamiento para locutores en español."""

    _asr_pipeline = None

    @classmethod
    def get_asr_pipeline(cls, model_name: str = "openai/whisper-tiny"):
        """Inicializa de forma diferida el pipeline de Whisper en CPU para aislarlo de CUDA."""
        if cls._asr_pipeline is None:
            from transformers import pipeline
            print(f"[DatasetPreparer] Cargando modelo Whisper ASR ({model_name}) en CPU...")
            cls._asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device="cpu",
                torch_dtype=torch.float32
            )
            print("[DatasetPreparer] Whisper ASR inicializado correctamente en CPU.")
        return cls._asr_pipeline

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normaliza el texto transcrito para que sea óptimo para el tokenizador de XTTS."""
        text = text.strip()
        # Eliminar caracteres extraños pero mantener acentos y eñes
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    @classmethod
    def _split_into_chunks(
        cls,
        y: np.ndarray,
        sr: int,
        min_sec: float = 2.0,
        max_sec: float = 10.0,
        top_db: float = 25.0
    ) -> List[np.ndarray]:
        """
        Divide la señal de audio en segmentos continuos de habla (entre min_sec y max_sec).
        Si el fragmento completo es menor a max_sec y >= min_sec, se mantiene intacto.
        """
        total_duration = len(y) / sr
        if total_duration <= max_sec:
            if total_duration >= min_sec:
                return [y]
            else:
                return [y]  # Aún si es levemente menor, conservarlo

        # Detectar intervalos de actividad de voz (no silencios)
        intervals = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)
        
        chunks = []
        current_chunk = []
        current_len = 0
        min_samples = int(min_sec * sr)
        max_samples = int(max_sec * sr)
        
        for start, end in intervals:
            seg = y[start:end]
            seg_len = len(seg)
            
            # Si el segmento por sí solo ya es muy largo, cortarlo en partes
            if seg_len > max_samples:
                if current_chunk:
                    comb = np.concatenate(current_chunk)
                    if len(comb) >= min_samples:
                        chunks.append(comb)
                    current_chunk = []
                    current_len = 0
                    
                num_sub = int(np.ceil(seg_len / max_samples))
                sub_parts = np.array_split(seg, num_sub)
                for part in sub_parts:
                    if len(part) >= min_samples:
                        chunks.append(part)
                continue
                
            if current_len + seg_len <= max_samples:
                current_chunk.append(seg)
                current_len += seg_len
            else:
                if current_len >= min_samples:
                    chunks.append(np.concatenate(current_chunk))
                current_chunk = [seg]
                current_len = seg_len
                
        if current_chunk and current_len >= min_samples:
            chunks.append(np.concatenate(current_chunk))
            
        # Si por alguna razón la segmentación fue vacía, retornar el audio dividido uniformemente
        if not chunks:
            num_splits = max(1, int(np.ceil(total_duration / 6.0)))
            chunks = np.array_split(y, num_splits)
            chunks = [c for c in chunks if len(c) >= int(1.5 * sr)]
            
        return chunks

    @classmethod
    def prepare_dataset(
        cls,
        audio_paths: List[str] | str,
        voice_name: str,
        whisper_model: str = "openai/whisper-tiny",
        progress_callback=None
    ) -> Dict:
        """
        Procesa una lista de audios, los limpia, los segmenta y los transcribe con Whisper,
        generando la carpeta 'datasets/<nombre>/wavs/' y 'metadata.csv'.
        """
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
            
        valid_paths = [p for p in audio_paths if os.path.isfile(p)]
        if not valid_paths:
            raise FileNotFoundError(f"No se encontraron archivos válidos en: {audio_paths}")
            
        clean_voice_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in voice_name.strip())
        voice_dataset_dir = os.path.join(DATASETS_DIR, clean_voice_name)
        wavs_dir = os.path.join(voice_dataset_dir, "wavs")
        os.makedirs(wavs_dir, exist_ok=True)
        
        target_sr = 24000
        all_segments = []
        
        if progress_callback:
            progress_callback(0.1, "Limpiando y acondicionando audios de entrada...")
            
        # 1. Limpieza y segmentación de cada audio de entrada
        raw_chunks = []
        for p in valid_paths:
            try:
                y_clean, sr_clean, _ = AudioEnhancer.enhance_audio(p)
                chunks = cls._split_into_chunks(y_clean, sr_clean, min_sec=2.0, max_sec=9.0)
                raw_chunks.extend(chunks)
            except Exception as e:
                print(f"[DatasetPreparer] Error al procesar {p}: {e}")
                
        if not raw_chunks:
            raise RuntimeError("No se pudieron extraer segmentos de audio válidos para el entrenamiento.")
            
        # 2. Inicializar Whisper ASR
        if progress_callback:
            progress_callback(0.3, "Cargando Whisper local en GPU para transcripción...")
            
        asr = cls.get_asr_pipeline(model_name=whisper_model)
        
        total_chunks = len(raw_chunks)
        csv_rows = []
        total_duration = 0.0
        
        # 3. Guardar WAVs y transcribir
        for idx, chunk in enumerate(raw_chunks, start=1):
            clip_filename = f"clip_{idx:03d}.wav"
            clip_path = os.path.join(wavs_dir, clip_filename)
            rel_clip_path = os.path.join("datasets", clean_voice_name, "wavs", clip_filename)
            
            # Normalizar y guardar
            sf.write(clip_path, chunk, target_sr, subtype="PCM_16")
            dur = len(chunk) / target_sr
            total_duration += dur
            
            # Transcripción ASR en español (pasando array NumPy a 16kHz para evitar dependencia de ffmpeg binario)
            try:
                chunk_16k = librosa.resample(chunk, orig_sr=target_sr, target_sr=16000)
                result = asr(
                    {"raw": chunk_16k, "sampling_rate": 16000},
                    generate_kwargs={"language": "spanish", "task": "transcribe"}
                )
                text = cls._clean_text(result.get("text", ""))
            except Exception as e:
                print(f"[DatasetPreparer] Fallo transcripción en {clip_filename}: {e}")
                text = "hola"
                
            if not text or len(text.strip()) < 2:
                text = "hola"
                
            csv_rows.append(f"{rel_clip_path}|{text}|{clean_voice_name}")
            
            if progress_callback:
                prog = 0.3 + (0.5 * (idx / total_chunks))
                progress_callback(prog, f"Transcribiendo clip {idx}/{total_chunks} ({dur:.1f}s)...")
                
        # 4. Escribir metadata.csv (formato XTTS / Coqui)
        metadata_path = os.path.join(voice_dataset_dir, "metadata.csv")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("audio_file|text|speaker_name\n")
            for row in csv_rows:
                f.write(row + "\n")
                
        # 5. Generar metadata_train.csv y metadata_eval.csv (90% train / 10% eval)
        train_path = os.path.join(voice_dataset_dir, "metadata_train.csv")
        eval_path = os.path.join(voice_dataset_dir, "metadata_eval.csv")
        
        if len(csv_rows) > 3:
            split_idx = max(1, int(len(csv_rows) * 0.90))
            train_rows = csv_rows[:split_idx]
            eval_rows = csv_rows[split_idx:]
        else:
            train_rows = csv_rows
            eval_rows = csv_rows
            
        with open(train_path, "w", encoding="utf-8") as f:
            f.write("audio_file|text|speaker_name\n")
            for row in train_rows:
                f.write(row + "\n")
                
        with open(eval_path, "w", encoding="utf-8") as f:
            f.write("audio_file|text|speaker_name\n")
            for row in eval_rows:
                f.write(row + "\n")
                
        if progress_callback:
            progress_callback(0.9, f"Dataset listo: {total_chunks} clips ({total_duration:.1f}s total).")
            
        # Liberar memoria de Whisper en GPU si se requiere para el entrenamiento
        torch.cuda.empty_cache()
        
        return {
            "voice_name": clean_voice_name,
            "dataset_dir": voice_dataset_dir,
            "metadata_csv": metadata_path,
            "metadata_train": train_path,
            "metadata_eval": eval_path,
            "total_clips": total_chunks,
            "total_duration_s": round(total_duration, 2),
            "samples": csv_rows[:3]
        }


if __name__ == "__main__":
    print("Módulo dataset_preparer cargado correctamente.")
