#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xtts_finetuner.py
-----------------
Motor de Micro Fine-Tuning de XTTS-v2 en GPU (NVIDIA RTX 5070):
- Ajuste fino de las capas GPT con AdamW y optimización de memoria (VRAM < 7 GB).
- Entrena sobre el dataset generado (metadata.csv y wavs/) durante 5 a 10 épocas.
- Guarda el checkpoint afinado en 'trained_models/<nombre_voz>/' con metadatos completos.
"""

import os
import sys
import time
import json
import shutil
from typing import Dict, List, Optional
import torch
import soundfile as sf

# Forzar UTF-8 en Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer, GPTTrainerConfig, GPTArgs
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "xtts_models")
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)


class XTTSFineTuner:
    """Motor de ajuste fino local para voces personalizadas con XTTS-v2."""

    @classmethod
    def parse_metadata_csv(cls, metadata_csv_path: str) -> List[Dict]:
        """Parsea el archivo metadata.csv delimitado por '|'."""
        if not os.path.isfile(metadata_csv_path):
            raise FileNotFoundError(f"No existe metadata.csv en: {metadata_csv_path}")
            
        samples = []
        with open(metadata_csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        header = lines[0].split("|") if lines else []
        for line in lines[1:]:
            parts = line.split("|")
            if len(parts) >= 2:
                rel_path = parts[0].strip()
                text = parts[1].strip()
                speaker = parts[2].strip() if len(parts) > 2 else "speaker"
                
                # Resolver ruta absoluta
                abs_audio_path = os.path.join(BASE_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path
                if os.path.isfile(abs_audio_path):
                    samples.append({
                        "audio_file": abs_audio_path,
                        "text": text,
                        "speaker_name": speaker,
                        "language": "es"
                    })
        return samples

    @classmethod
    def train_voice(
        cls,
        voice_name: str,
        dataset_dir: str,
        epochs: int = 6,
        batch_size: int = 2,
        lr: float = 5e-6,
        progress_callback=None
    ) -> Dict:
        """
        Ejecuta el ciclo de micro fine-tuning de XTTS-v2 en GPU CUDA:
        1. Carga configuración base y pesos de XTTS-v2.
        2. Configura GPTTrainer y DataLoader con XTTSDataset.
        3. Ejecuta bucle de entrenamiento AdamW en GPU.
        4. Guarda el modelo afinado en 'trained_models/<nombre_voz>/'.
        """
        t_start = time.time()
        clean_voice_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in voice_name.strip())
        output_model_dir = os.path.join(TRAINED_MODELS_DIR, clean_voice_name)
        os.makedirs(output_model_dir, exist_ok=True)
        
        metadata_csv = os.path.join(dataset_dir, "metadata.csv")
        samples = cls.parse_metadata_csv(metadata_csv)
        if not samples:
            raise ValueError(f"No se encontraron clips válidos en {metadata_csv}")
            
        if progress_callback:
            progress_callback(0.05, "Configurando entorno de entrenamiento en GPU...")
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Configurar GPTTrainerConfig
        base_config_path = os.path.join(MODELS_DIR, "config.json")
        vocab_path = os.path.join(MODELS_DIR, "vocab.json")
        xtts_checkpoint = os.path.join(MODELS_DIR, "model.pth")
        dvae_checkpoint = os.path.join(MODELS_DIR, "dvae.pth")
        mel_norms_path = os.path.join(MODELS_DIR, "mel_stats.pth")
        
        trainer_config = GPTTrainerConfig()
        if os.path.exists(base_config_path):
            trainer_config.load_json(base_config_path)
            
        trainer_config.lr = float(lr)
        trainer_config.model_args.tokenizer_file = vocab_path
        trainer_config.model_args.xtts_checkpoint = xtts_checkpoint
        trainer_config.model_args.dvae_checkpoint = dvae_checkpoint
        if os.path.exists(mel_norms_path):
            trainer_config.model_args.mel_norm_file = mel_norms_path
            
        # 2. Inicializar GPTTrainer
        if progress_callback:
            progress_callback(0.15, "Inicializando pesos base de XTTS-v2 en GPU...")
            
        trainer_model = GPTTrainer(trainer_config)
        trainer_model.to(device)
        
        # Modo de entrenamiento para GPT, resto en eval
        trainer_model.eval()
        trainer_model.xtts.gpt.train()
        
        # 3. Dataset y DataLoader
        dataset = XTTSDataset(
            config=trainer_config,
            samples=samples,
            tokenizer=trainer_model.xtts.tokenizer,
            sample_rate=trainer_config.audio.sample_rate,
            is_eval=False
        )
        
        actual_batch_size = min(batch_size, len(samples))
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            drop_last=False
        )
        
        # 4. Optimizador AdamW solo para GPT
        optimizer = torch.optim.AdamW(
            trainer_model.xtts.gpt.parameters(),
            lr=lr,
            weight_decay=0.01,
            eps=1e-8
        )
        
        total_steps = epochs * len(data_loader)
        step = 0
        avg_loss = 0.0
        
        if progress_callback:
            progress_callback(0.20, f"Iniciando {epochs} épocas de entrenamiento ({len(samples)} clips)...")
            
        # 5. Bucle de Entrenamiento
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in data_loader:
                step += 1
                optimizer.zero_grad()
                
                # Mover batch a GPU y formatear
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch = trainer_model.format_batch_on_device(batch)
                
                _, loss_dict = trainer_model.train_step(batch, criterion=None)
                loss = loss_dict["loss"]
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer_model.xtts.gpt.parameters(), max_norm=1.0)
                optimizer.step()
                
                loss_val = loss.item()
                epoch_loss += loss_val
                num_batches += 1
                
                if progress_callback:
                    current_prog = 0.20 + (0.65 * (step / total_steps))
                    progress_callback(
                        current_prog,
                        f"Época {epoch}/{epochs} | Batch {num_batches}/{len(data_loader)} | Loss: {loss_val:.4f}"
                    )
                    
            avg_loss = epoch_loss / max(1, num_batches)
            print(f"[XTTSFineTuner] Época {epoch}/{epochs} completada. Loss promedio: {avg_loss:.4f}")
            
        # 6. Extraer latentes acústicos del locutor con el modelo afinado
        if progress_callback:
            progress_callback(0.88, "Extrayendo vectores latentes del locutor...")
            
        audio_files = [s["audio_file"] for s in samples]
        gpt_latent, spk_emb = trainer_model.xtts.get_conditioning_latents(
            audio_path=audio_files,
            gpt_cond_len=30,
            max_ref_length=60,
            sound_norm_refs=True
        )
        
        # 7. Guardar Checkpoint y Archivos de Inferencia
        if progress_callback:
            progress_callback(0.92, "Guardando modelo ajustado y artefactos...")
            
        out_model_path = os.path.join(output_model_dir, "model.pth")
        out_spk_path = os.path.join(output_model_dir, "speaker_embedding.pt")
        out_vocab_path = os.path.join(output_model_dir, "vocab.json")
        out_config_path = os.path.join(output_model_dir, "config.json")
        out_meta_path = os.path.join(output_model_dir, "metadata.json")
        
        # Guardar pesos ajustados GPT del modelo (~350 MB)
        gpt_state_dict = trainer_model.xtts.gpt.state_dict()
        torch.save({"gpt": gpt_state_dict}, out_model_path)
        
        # Guardar embedding del locutor
        torch.save({
            "gpt_cond_latent": gpt_latent.cpu(),
            "speaker_embedding": spk_emb.cpu(),
            "speaker_name": clean_voice_name
        }, out_spk_path)
        
        # Copiar vocab.json y config.json
        if os.path.exists(vocab_path):
            shutil.copy2(vocab_path, out_vocab_path)
            
        config_data = trainer_config.to_dict()
        with open(out_config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
            
        elapsed = time.time() - t_start
        model_size_mb = os.path.getsize(out_model_path) / (1024 * 1024)
        
        meta_info = {
            "voice_name": clean_voice_name,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "final_loss": round(avg_loss, 4),
            "training_duration_s": round(elapsed, 2),
            "num_training_clips": len(samples),
            "model_size_mb": round(model_size_mb, 2),
            "device": device
        }
        with open(out_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=2)
            
        # Liberar memoria de GPU
        del data_loader
        del dataset
        del optimizer
        del trainer_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        if progress_callback:
            progress_callback(1.0, f"¡Entrenamiento completado con éxito en {elapsed:.1f}s! (Loss: {avg_loss:.4f})")
            
        return {
            "voice_name": clean_voice_name,
            "model_dir": output_model_dir,
            "model_path": out_model_path,
            "speaker_path": out_spk_path,
            "epochs": epochs,
            "final_loss": round(avg_loss, 4),
            "duration_s": round(elapsed, 2),
            "model_size_mb": round(model_size_mb, 2)
        }


if __name__ == "__main__":
    print("Módulo xtts_finetuner cargado correctamente.")
