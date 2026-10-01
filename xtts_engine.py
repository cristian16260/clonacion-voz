#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xtts_engine.py
--------------
Motor central de clonación e inferencia de voz en español basado en Coqui XTTS-v2.
- Soporta aceleración GPU (CUDA) en NVIDIA RTX 5070.
- Extrae perfiles de voz ultra ligeros (~130 KB) en archivos .pt (gpt_cond_latent + speaker_embedding).
- Inferencia directa de alta fidelidad en español y streaming de chunks.
"""

import io
import os
import sys
import time
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

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from audio_enhancer import AudioEnhancer, AudioEnhancementConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "xtts_models")
VOICES_DIR = os.path.join(BASE_DIR, "voices")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, "trained_models")

os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)


class XTTSEngine:
    _instance = None

    def __init__(self, model_dir: str = MODELS_DIR):
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._current_finetuned_model = None
        print(f"[XTTSEngine] Inicializando en dispositivo: {self.device.upper()}")
        
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No se encontró 'config.json' en {self.model_dir}. Verifique que el modelo esté descargado.")
            
        self.config = XttsConfig()
        self.config.load_json(config_path)
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=self.model_dir, eval=True)
        
        if self.device == "cuda":
            self.model.cuda()
            
        self.sample_rate = 24000
        print("[XTTSEngine] Modelo XTTS-v2 cargado exitosamente.")

    @classmethod
    def get_instance(cls, model_dir: str = MODELS_DIR):
        if cls._instance is None:
            cls._instance = cls(model_dir=model_dir)
        return cls._instance

    @staticmethod
    def get_available_trained_models() -> list[str]:
        """Devuelve la lista de modelos entrenados (fine-tuned) disponibles en trained_models/."""
        if not os.path.isdir(TRAINED_MODELS_DIR):
            return []
        models = []
        for d in os.listdir(TRAINED_MODELS_DIR):
            if d == "vits":
                continue
            sub = os.path.join(TRAINED_MODELS_DIR, d)
            if os.path.isdir(sub) and (os.path.isfile(os.path.join(sub, "model.pth")) or os.path.isfile(os.path.join(sub, "speaker_embedding.pt"))):
                models.append(d)
        return sorted(models)

    def delete_voice_profile(self, voice_name: str) -> bool:
        """Elimina un perfil .pt o carpeta de modelo entrenado de XTTS."""
        clean_name = os.path.splitext(os.path.basename(voice_name))[0]
        deleted = False
        
        # 1. Perfil .pt
        pt_path = os.path.join(VOICES_DIR, f"{clean_name}.pt")
        if os.path.isfile(pt_path):
            try:
                os.remove(pt_path)
                deleted = True
            except Exception:
                pass
            
        # 2. Modelo afinado en trained_models/
        model_dir = os.path.join(TRAINED_MODELS_DIR, clean_name)
        if os.path.isdir(model_dir) and clean_name != "vits":
            try:
                import shutil
                shutil.rmtree(model_dir, ignore_errors=True)
                if self._current_finetuned_model == clean_name:
                    self._current_finetuned_model = None
                deleted = True
            except Exception:
                pass
            
        return deleted

    def load_finetuned_model(self, model_name: str) -> bool:
        """Carga los pesos GPT afinados de un modelo específico en GPU."""
        clean_name = os.path.splitext(os.path.basename(model_name))[0]
        model_dir = os.path.join(TRAINED_MODELS_DIR, clean_name)
        model_pth = os.path.join(model_dir, "model.pth")
        
        if not os.path.isfile(model_pth):
            raise FileNotFoundError(f"No se encontró el checkpoint entrenado en: {model_pth}")
            
        if self._current_finetuned_model == clean_name:
            return True
            
        print(f"[XTTSEngine] Cargando checkpoint afinado: '{clean_name}'...")
        ckpt = torch.load(model_pth, map_location=self.device, weights_only=False)
        
        if "gpt" in ckpt:
            self.model.gpt.load_state_dict(ckpt["gpt"], strict=False)
        elif "model" in ckpt:
            gpt_dict = {k.replace("gpt.", ""): v for k, v in ckpt["model"].items() if "gpt." in k}
            if gpt_dict:
                self.model.gpt.load_state_dict(gpt_dict, strict=False)
            else:
                self.model.gpt.load_state_dict(ckpt["model"], strict=False)
        else:
            self.model.gpt.load_state_dict(ckpt, strict=False)
            
        self.model.gpt.init_gpt_for_inference(kv_cache=True, use_deepspeed=False)
        self.model.gpt.eval()
        self._current_finetuned_model = clean_name
        print(f"[XTTSEngine] Checkpoint afinado '{clean_name}' cargado exitosamente.")
        return True

    def restore_base_model(self) -> bool:
        """Restaura los pesos base originales de XTTS-v2 si un modelo afinado estaba cargado."""
        if self._current_finetuned_model is None:
            return True
            
        print("[XTTSEngine] Restaurando pesos base originales de XTTS-v2...")
        self.model.load_checkpoint(self.config, checkpoint_dir=self.model_dir, eval=True)
        if self.device == "cuda":
            self.model.cuda()
        self._current_finetuned_model = None
        print("[XTTSEngine] Modelo base restaurado.")
        return True

    def extract_voice_profile(
        self,
        audio_paths: str | list[str],
        voice_name: str,
        temperature: float = 0.60,
        speed: float = 1.0,
        enhance_audio: bool = True
    ) -> tuple[str, float]:
        """
        Extrae el perfil de voz (gpt_cond_latent + speaker_embedding) a partir de 1 o múltiples
        archivos de audio (MP3/WAV) del mismo locutor y los guarda en 'voices/<nombre>.pt',
        junto con su calibración de temperatura y fidelidad para SageMaker/producción.
        Opcionalmente aplica preprocesamiento acústico con AudioEnhancer (denoise, VAD trim y normalización).
        Retorna (ruta_archivo_pt, duracion_total_referencia).
        """
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
            
        valid_paths = [p for p in audio_paths if os.path.isfile(p)]
        if not valid_paths:
            raise FileNotFoundError(f"No se encontraron archivos de audio válidos en: {audio_paths}")
            
        clean_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in voice_name.strip())
        out_pt_path = os.path.join(VOICES_DIR, f"{clean_name}.pt")
        
        # Procesar con AudioEnhancer si está activado
        processed_paths = []
        if enhance_audio:
            for p in valid_paths:
                try:
                    _, _, clean_path = AudioEnhancer.enhance_audio(p)
                    processed_paths.append(clean_path)
                except Exception as e:
                    print(f"[XTTSEngine] Advertencia: Fallo al limpiar {p}: {e}. Usando original.")
                    processed_paths.append(p)
        else:
            processed_paths = valid_paths
        
        # Obtener latentes de condicionamiento combinando todos los clips (hasta 60s total)
        gpt_latent, spk_emb = self.model.get_conditioning_latents(
            audio_path=processed_paths,
            gpt_cond_len=30,
            max_ref_length=60,
            sound_norm_refs=True
        )
        
        total_duration = 0.0
        for p in processed_paths:
            try:
                info = sf.info(p)
                total_duration += info.duration
            except Exception:
                pass
        
        profile_data = {
            "gpt_cond_latent": gpt_latent.cpu(),
            "speaker_embedding": spk_emb.cpu(),
            "metadata": {
                "voice_name": voice_name.strip(),
                "calibrated_temperature": float(temperature),
                "calibrated_speed": float(speed),
                "enhanced": bool(enhance_audio),
                "reference_files": [os.path.basename(p) for p in valid_paths],
                "total_reference_duration_s": round(total_duration, 2),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        torch.save(profile_data, out_pt_path)
        return out_pt_path, total_duration

    def delete_voice_profile(self, voice_identifier: str) -> bool:
        """
        Elimina un perfil de voz .pt existente o un modelo entrenado en trained_models/
        junto con sus datasets de entrenamiento asociados en datasets/.
        Retorna True si fue eliminado, False si no existía.
        """
        clean_name = os.path.splitext(os.path.basename(voice_identifier))[0]
        deleted = False
        
        # 1. Si era el modelo actualmente cargado en memoria, restaurar base
        if self._current_finetuned_model == clean_name:
            self.restore_base_model()
            
        # 2. Eliminar modelo afinado en trained_models/
        trained_dir = os.path.join(TRAINED_MODELS_DIR, clean_name)
        if os.path.isdir(trained_dir):
            shutil.rmtree(trained_dir, ignore_errors=True)
            deleted = True
            
        # 3. Eliminar dataset de entrenamiento en datasets/
        ds_dir = os.path.join(BASE_DIR, "datasets", clean_name)
        if os.path.isdir(ds_dir):
            shutil.rmtree(ds_dir, ignore_errors=True)
            deleted = True
            
        # 4. Eliminar perfil .pt en voices/
        pt_path = os.path.join(VOICES_DIR, f"{clean_name}.pt")
        if os.path.isfile(pt_path):
            os.remove(pt_path)
            deleted = True
            
        return deleted

    def load_voice_profile(self, voice_identifier: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Carga un perfil de voz guardado (.pt), un modelo entrenado en trained_models/
        o procesa un audio de referencia directamente.
        Retorna (gpt_cond_latent, speaker_embedding, metadata).
        """
        # 1. Verificar si es un modelo entrenado en trained_models/
        clean_name = os.path.splitext(os.path.basename(voice_identifier))[0]
        trained_dir = os.path.join(TRAINED_MODELS_DIR, clean_name)
        if os.path.isdir(trained_dir):
            self.load_finetuned_model(clean_name)
            spk_file = os.path.join(trained_dir, "speaker_embedding.pt")
            if os.path.isfile(spk_file):
                data = torch.load(spk_file, map_location=self.device, weights_only=False)
                gpt_latent = data["gpt_cond_latent"].to(self.device)
                spk_emb = data["speaker_embedding"].to(self.device)
                meta_json = os.path.join(trained_dir, "metadata.json")
                meta = {}
                if os.path.isfile(meta_json):
                    try:
                        with open(meta_json, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                meta["is_finetuned"] = True
                meta["voice_name"] = clean_name
                meta["calibrated_temperature"] = meta.get("calibrated_temperature", 0.55)
                meta["calibrated_speed"] = meta.get("calibrated_speed", 1.0)
                return gpt_latent, spk_emb, meta

        # 2. Si se estaba usando un modelo entrenado y ahora se pide un perfil normal, restaurar base
        if self._current_finetuned_model is not None:
            self.restore_base_model()

        # 3. Si es un nombre en la carpeta voices/
        candidate_pt = os.path.join(VOICES_DIR, f"{voice_identifier}.pt")
        if os.path.isfile(candidate_pt):
            profile_path = candidate_pt
        elif os.path.isfile(voice_identifier) and voice_identifier.endswith(".pt"):
            profile_path = voice_identifier
        elif os.path.isfile(voice_identifier):
            # Es un archivo de audio directo (WAV/MP3)
            gpt_latent, spk_emb = self.model.get_conditioning_latents(
                audio_path=[voice_identifier],
                gpt_cond_len=30,
                max_ref_length=60,
                sound_norm_refs=True
            )
            return gpt_latent.to(self.device), spk_emb.to(self.device), {"name": os.path.basename(voice_identifier)}
        else:
            raise FileNotFoundError(f"No se encontró el perfil de voz, modelo entrenado ni audio para: '{voice_identifier}'")

        data = torch.load(profile_path, map_location=self.device, weights_only=False)
        gpt_latent = data["gpt_cond_latent"].to(self.device)
        spk_emb = data["speaker_embedding"].to(self.device)
        meta = data.get("metadata", {})
        return gpt_latent, spk_emb, meta

    def synthesize(
        self,
        text: str,
        voice_identifier: str,
        output_path: str = None,
        language: str = "es",
        speed: float = None,
        temperature: float = None,
        repetition_penalty: float = 5.0
    ) -> dict:
        """
        Sintetiza texto en español utilizando el perfil de voz clonada.
        Aplica automáticamente la temperatura y velocidad calibradas en el perfil .pt
        a menos que se especifiquen valores manuales.
        """
        if not text or not text.strip():
            raise ValueError("El texto a sintetizar no puede estar vacío.")

        gpt_latent, spk_emb, meta = self.load_voice_profile(voice_identifier)
        
        # Leer valores calibrados del perfil
        actual_temp = float(temperature) if temperature is not None else float(meta.get("calibrated_temperature", 0.60))
        actual_speed = float(speed) if speed is not None else float(meta.get("calibrated_speed", 1.0))
        
        if output_path is None:
            ts = int(time.time())
            safe_id = "".join(c if c.isalnum() else "_" for c in str(voice_identifier))
            output_path = os.path.join(OUTPUTS_DIR, f"synth_{safe_id}_{ts}.wav")

        t_start = time.perf_counter()
        
        with torch.inference_mode():
            out = self.model.inference(
                text=text.strip(),
                language=language,
                gpt_cond_latent=gpt_latent,
                speaker_embedding=spk_emb,
                temperature=actual_temp,
                length_penalty=1.0,
                repetition_penalty=repetition_penalty,
                top_k=50,
                top_p=0.85,
                speed=actual_speed
            )
            
        t_end = time.perf_counter()
        
        wav = np.array(out["wav"], dtype=np.float32)
        sf.write(output_path, wav, self.sample_rate)
        
        duration_s = len(wav) / self.sample_rate
        infer_time_s = t_end - t_start
        rtf = infer_time_s / duration_s if duration_s > 0 else 0.0

        return {
            "output_path": output_path,
            "duration_s": duration_s,
            "infer_time_s": infer_time_s,
            "rtf": rtf,
            "sample_rate": self.sample_rate,
            "metadata": meta
        }

    def synthesize_stream(
        self,
        text: str,
        voice_identifier: str,
        language: str = "es",
        speed: float = None,
        temperature: float = None,
        stream_chunk_size: int = 20
    ):
        """
        Generador para streaming en tiempo real por fragmentos (chunks).
        Aplica automáticamente los parámetros de temperatura y velocidad calibrados del perfil .pt.
        Yields trozos de audio (numpy array float32).
        """
        gpt_latent, spk_emb, meta = self.load_voice_profile(voice_identifier)
        
        actual_temp = float(temperature) if temperature is not None else float(meta.get("calibrated_temperature", 0.60))
        actual_speed = float(speed) if speed is not None else float(meta.get("calibrated_speed", 1.0))
        
        chunks = self.model.inference_stream(
            text=text.strip(),
            language=language,
            gpt_cond_latent=gpt_latent,
            speaker_embedding=spk_emb,
            temperature=actual_temp,
            speed=actual_speed,
            stream_chunk_size=stream_chunk_size
        )
        
        for chunk in chunks:
            if isinstance(chunk, torch.Tensor):
                yield chunk.cpu().numpy().astype(np.float32)
            else:
                yield np.array(chunk, dtype=np.float32)
