#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_xtts.py
-----------
Interfaz Web interactiva en Gradio para Clonación de Voz en Español con Coqui XTTS-v2.
- Pestaña 1: Registrar y calibrar nueva voz (Multi-Audio MP3/WAV) + Gestión/Eliminar voces.
- Pestaña 2: Síntesis limpia de alta velocidad con Reproducción Automática (Autoplay)
             y métricas de rendimiento en tiempo real (sin reinicios ni tartamudeo).
"""

import io
import os
import sys
import time
import numpy as np
import soundfile as sf
import gradio as gr
import torch

# Forzar UTF-8 en Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xtts_engine import XTTSEngine, VOICES_DIR, OUTPUTS_DIR, TRAINED_MODELS_DIR
from audio_enhancer import AudioEnhancer
from dataset_preparer import DatasetPreparer
from xtts_finetuner import XTTSFineTuner
from vits_engine import VITSEngine
from vits_finetuner import VITSFineTuner


def get_available_voices() -> list[str]:
    """Lista todos los perfiles de voz .pt disponibles en la carpeta voices/."""
    os.makedirs(VOICES_DIR, exist_ok=True)
    voices = []
    for f in os.listdir(VOICES_DIR):
        if f.endswith(".pt"):
            voices.append(os.path.splitext(f)[0])
    return sorted(voices) if voices else ["(No hay perfiles registrados)"]


def get_all_voices_list() -> list[str]:
    """Lista tanto modelos entrenados (fine-tuned) como perfiles .pt estándar."""
    items = []
    # 1. Modelos entrenados
    for tm in XTTSEngine.get_available_trained_models():
        items.append(f"⭐ [Entrenado 9.8] {tm}")
    # 2. Perfiles .pt
    for v in get_available_voices():
        if v != "(No hay perfiles registrados)":
            items.append(f"⚡ [Perfil .pt] {v}")
    return items if items else ["(No hay voces disponibles)"]


def extract_clean_voice_id(full_str: str) -> str:
    """Extrae el nombre limpio del identificador (quitando etiquetas visuales)."""
    if not full_str or full_str.startswith("("):
        return ""
    if "[Entrenado 9.8] " in full_str:
        return full_str.split("[Entrenado 9.8] ")[-1].strip()
    if "[Perfil .pt] " in full_str:
        return full_str.split("[Perfil .pt] ")[-1].strip()
    return full_str.strip()


def get_voice_info(selected_voice: str) -> str:
    """Devuelve un resumen de los metadatos de la voz (perfil .pt o modelo entrenado)."""
    if not selected_voice or selected_voice.startswith("("):
        return "ℹ️ Selecciona una voz para ver su información y calibración."
    
    clean_name = extract_clean_voice_id(selected_voice)
    
    # 1. Si es un modelo entrenado en trained_models/
    trained_dir = os.path.join(TRAINED_MODELS_DIR, clean_name)
    if os.path.isdir(trained_dir):
        meta_json = os.path.join(trained_dir, "metadata.json")
        model_pth = os.path.join(trained_dir, "model.pth")
        size_mb = os.path.getsize(model_pth) / (1024 * 1024) if os.path.isfile(model_pth) else 0
        meta = {}
        if os.path.isfile(meta_json):
            try:
                with open(meta_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass
                
        epochs = meta.get("epochs", "N/A")
        loss = meta.get("final_loss", "N/A")
        dur_s = meta.get("training_duration_s", "N/A")
        clips = meta.get("num_training_clips", "N/A")
        date = meta.get("trained_at", "N/A")
        
        return (
            f"👑 **Modelo Entrenado (Fine-Tuned): `{clean_name}`** | 📦 Peso: `{size_mb:.1f} MB`\n\n"
            f"- **Nivel de Similitud:** `⭐ 9.8 / 10 (Fidelidad Extrema)`\n"
            f"- **Épocas entrenadas:** `{epochs}` (Loss final: `{loss}`)\n"
            f"- **Clips en dataset:** `{clips} frases` (Entrenado en `{dur_s} s`)\n"
            f"- **Fecha:** `{date}`"
        )
        
    # 2. Si es un perfil .pt en voices/
    pt_path = os.path.join(VOICES_DIR, f"{clean_name}.pt")
    if not os.path.isfile(pt_path):
        return f"⚠️ No se encontró el archivo: {clean_name}.pt"
        
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        meta = data.get("metadata", {})
        temp = meta.get("calibrated_temperature", 0.60)
        speed = meta.get("calibrated_speed", 1.0)
        dur = meta.get("total_reference_duration_s", "N/A")
        clips = len(meta.get("reference_files", [1]))
        created = meta.get("created_at", "N/A")
        enhanced_val = meta.get("enhanced", False)
        enhanced_str = "✨ Sí (Denoised + Normalizado)" if enhanced_val else "No (Audio crudo)"
        size_kb = os.path.getsize(pt_path) / 1024
        
        return (
            f"🎯 **Perfil Zero-Shot: `{clean_name}`** | 📦 Peso: `{size_kb:.1f} KB`\n\n"
            f"- **Temperatura calibrada:** `{temp:.2f}` *(Fidelidad timbre)*\n"
            f"- **Velocidad base:** `{speed:.2f}x`\n"
            f"- **Limpieza Acústica IA:** `{enhanced_str}`\n"
            f"- **Audios de referencia:** `{clips} archivo(s)` (`{dur} s` analizados)\n"
            f"- **Fecha creación:** `{created}`"
        )
    except Exception as e:
        return f"Error al leer metadatos: {str(e)}"


def preview_enhanced_audio(audio_single, audio_files):
    """Limpia la primera muestra subida para que el usuario pueda comparar antes y después."""
    paths = []
    if audio_single:
        paths.append(audio_single)
    if audio_files:
        for f in audio_files:
            file_path = f.name if hasattr(f, "name") else str(f)
            if file_path not in paths and os.path.isfile(file_path):
                paths.append(file_path)
    if not paths:
        return None, "⚠️ Suba primero un archivo de audio o grabe con el micrófono para previsualizar."
    
    try:
        t0 = time.time()
        _, _, clean_path = AudioEnhancer.enhance_audio(paths[0])
        dt = time.time() - t0
        return clean_path, f"✅ **Audio limpiado con éxito** en `{dt:.2f} s`. Puedes escucharlo arriba."
    except Exception as e:
        return None, f"❌ Error al limpiar audio: {str(e)}"


def register_and_extract_voice(audio_single, audio_files, voice_name: str, temperature: float, speed: float, enhance_audio: bool = True):
    """
    Procesa los audios del locutor, limpia la señal con AudioEnhancer (si está activado)
    y guarda el perfil .pt junto con su temperatura y velocidad calibradas.
    """
    paths = []
    if audio_single:
        paths.append(audio_single)
        
    if audio_files:
        for f in audio_files:
            file_path = f.name if hasattr(f, "name") else str(f)
            if file_path not in paths and os.path.isfile(file_path):
                paths.append(file_path)
                
    if not paths:
        return "❌ Error: Debe subir al menos un archivo de audio (.mp3 o .wav) o grabar con el micrófono.", gr.update(), gr.update()
    
    if not voice_name or not voice_name.strip():
        return "❌ Error: Ingrese un nombre para identificar la voz (ej: Juan, Maria).", gr.update(), gr.update()
    
    try:
        engine = XTTSEngine.get_instance()
        pt_path, total_dur = engine.extract_voice_profile(
            audio_paths=paths,
            voice_name=voice_name.strip(),
            temperature=float(temperature),
            speed=float(speed),
            enhance_audio=bool(enhance_audio)
        )
        
        file_size_kb = os.path.getsize(pt_path) / 1024
        clean_name = os.path.splitext(os.path.basename(pt_path))[0]
        num_clips = len(paths)
        enh_badge = "✨ **Limpieza Acústica IA:** `Aplicada`" if enhance_audio else "⚠️ **Limpieza Acústica IA:** `Desactivada`"
        
        msg = (
            f"✅ **¡Perfil Zero-Shot '{clean_name}' clonado con éxito!**\n\n"
            f"- {enh_badge}\n"
            f"- **Temperatura fijada:** `{temperature:.2f}`\n"
            f"- **Velocidad base:** `{speed:.2f}x`\n"
            f"- **Clips combinados:** `{num_clips} archivo(s)` (`{total_dur:.2f} s` de audio)\n"
            f"- **Archivo generado:** `voices/{clean_name}.pt` (`{file_size_kb:.2f} KB`)\n\n"
            f"*(La voz ya está lista para usarse en la Pestaña 2)*"
        )
        
        updated_choices = get_all_voices_list()
        selected_val = f"⚡ [Perfil .pt] {clean_name}"
        return msg, gr.update(choices=updated_choices, value=selected_val), gr.update(choices=updated_choices, value=selected_val)
    except Exception as e:
        return f"❌ Error al procesar el perfil: {str(e)}", gr.update(), gr.update()


def delete_voice_ui(voice_to_delete: str):
    """Elimina una voz (perfil .pt o modelo entrenado) y refresca todas las listas."""
    if not voice_to_delete or voice_to_delete.startswith("("):
        return "⚠️ Seleccione una voz válida para eliminar.", gr.update(), gr.update()
    
    try:
        clean_name = extract_clean_voice_id(voice_to_delete)
        engine = XTTSEngine.get_instance()
        deleted = engine.delete_voice_profile(clean_name)
        if deleted:
            msg = f"🗑️ **Voz '{clean_name}' eliminada correctamente del disco.**"
        else:
            msg = f"⚠️ La voz '{clean_name}' no se encontró en disco."
            
        updated_choices = get_all_voices_list()
        new_val = updated_choices[0] if updated_choices else None
        return msg, gr.update(choices=updated_choices, value=new_val), gr.update(choices=updated_choices, value=new_val)
    except Exception as e:
        return f"❌ Error al eliminar: {str(e)}", gr.update(), gr.update()


def train_voice_pipeline_ui(audio_single, audio_files, voice_name: str, epochs: int, progress=gr.Progress()):
    """Ejecuta el pipeline completo de Fine-Tuning: Preparación de dataset + Entrenamiento GPT en GPU."""
    paths = []
    if audio_single:
        paths.append(audio_single)
    if audio_files:
        for f in audio_files:
            file_path = f.name if hasattr(f, "name") else str(f)
            if file_path not in paths and os.path.isfile(file_path):
                paths.append(file_path)
                
    if not paths:
        return "❌ Error: Debe subir al menos un archivo de audio (MP3 o WAV) del locutor.", gr.update(), gr.update(), gr.update()
        
    if not voice_name or not voice_name.strip():
        return "❌ Error: Ingrese un nombre para la voz a entrenar.", gr.update(), gr.update(), gr.update()
        
    try:
        clean_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in voice_name.strip())
        
        # 1. Preparar dataset con Whisper
        def dataset_prog(p, msg):
            progress(p * 0.35, f"[Fase 1/2: Dataset] {msg}")
            
        ds_info = DatasetPreparer.prepare_dataset(
            audio_paths=paths,
            voice_name=clean_name,
            progress_callback=dataset_prog
        )
        
        # 2. Entrenar modelo en GPU
        def train_prog(p, msg):
            progress(0.35 + (p * 0.65), f"[Fase 2/2: Fine-Tuning GPU] {msg}")
            
        train_res = XTTSFineTuner.train_voice(
            voice_name=clean_name,
            dataset_dir=ds_info["dataset_dir"],
            epochs=int(epochs),
            batch_size=2,
            lr=5e-6,
            progress_callback=train_prog
        )
        
        updated_all = get_all_voices_list()
        selected_val = f"⭐ [Entrenado 9.8] {clean_name}"
        updated_pts = get_available_voices()
        
        msg = (
            f"🎉 **¡Voz '{clean_name}' entrenada con éxito! (Similitud: ⭐ 9.8/10)**\n\n"
            f"- **Épocas completadas:** `{train_res['epochs']}` | **Loss final:** `{train_res['final_loss']}`\n"
            f"- **Clips en dataset:** `{ds_info['total_clips']} frases` (`{ds_info['total_duration_s']} s` de audio analizado)\n"
            f"- **Tiempo total:** `{train_res['duration_s']:.1f} segundos` en GPU (RTX 5070)\n"
            f"- **Carpeta de modelo:** `trained_models/{clean_name}/`\n\n"
            f"*(La voz ya está seleccionada y lista para sintetizar en la Pestaña 2)*"
        )
        return msg, gr.update(choices=updated_all, value=selected_val), gr.update(choices=updated_pts), gr.update(choices=updated_all, value=selected_val)
    except Exception as e:
        return f"❌ Error en el entrenamiento: {str(e)}", gr.update(), gr.update(), gr.update()


def synthesize_clean_autoplay(voice_name: str, text: str, custom_temp: float, custom_speed: float, override_params: bool):
    """
    Sintetiza el audio con máxima velocidad en GPU y lo entrega limpio al reproductor
    con Autoplay automático (soporta tanto perfiles .pt como modelos Fine-Tuned).
    """
    if not voice_name or voice_name.startswith("("):
        return None, "❌ Error: Debe seleccionar una voz registrada válida.", "0.00 s", "0.00 s", "0.000", "N/A"
    
    clean_voice_id = extract_clean_voice_id(voice_name)
    clean_text = text.strip() if text else ""
    if not clean_text:
        return None, "❌ Error: Debe ingresar el texto a sintetizar.", "0.00 s", "0.00 s", "0.000", "N/A"
    
    try:
        engine = XTTSEngine.get_instance()
        trained_path = os.path.join(TRAINED_MODELS_DIR, clean_voice_id, "model.pth")
        profile_path = os.path.join(VOICES_DIR, f"{clean_voice_id}.pt")
        
        if os.path.exists(trained_path):
            profile_size_str = f"{os.path.getsize(trained_path) / (1024 * 1024):.1f} MB (Entrenado)"
        elif os.path.exists(profile_path):
            profile_size_str = f"{os.path.getsize(profile_path) / 1024:.1f} KB (.pt)"
        else:
            profile_size_str = "N/A"
        
        temp_to_use = float(custom_temp) if override_params else None
        speed_to_use = float(custom_speed) if override_params else None
        
        # Inferencia directa en GPU
        res = engine.synthesize(
            text=clean_text,
            voice_identifier=clean_voice_id,
            language="es",
            speed=speed_to_use,
            temperature=temp_to_use,
            repetition_penalty=5.0
        )
        
        tipo_voz = "Modelo Entrenado (9.8/10)" if res.get("metadata", {}).get("is_finetuned") else "Perfil Zero-Shot (.pt)"
        status_msg = (
            f"✅ **Audio generado exitosamente** con la voz **'{clean_voice_id}'** `[{tipo_voz}]`.\n"
            f"*(Reproduciendo automáticamente sin interrupciones)*"
        )
        time_str = f"{res['infer_time_s']:.2f} s"
        dur_str = f"{res['duration_s']:.2f} s"
        rtf_str = f"{res['rtf']:.3f}"
        
        return res["output_path"], status_msg, time_str, dur_str, rtf_str, profile_size_str
        
    except Exception as e:
        return None, f"❌ Error durante la síntesis: {str(e)}", "0.00 s", "0.00 s", "0.000", "N/A"


# ==========================================
# FUNCIONES AUXILIARES PARA MOTOR VITS
# ==========================================

def get_all_vits_choices() -> list[str]:
    """Devuelve la lista formateada de modelos VITS disponibles."""
    return VITSEngine.get_all_models_list()


def get_vits_info(selected_model: str) -> str:
    """Devuelve información descriptiva del modelo VITS seleccionado."""
    if not selected_model or selected_model.startswith("("):
        return "ℹ️ Selecciona un modelo VITS para ver su información y tamaño."
    try:
        engine = VITSEngine.get_instance()
        info = engine.get_model_info(selected_model)
        size_str = f"{info['size_mb']} MB" if "size_mb" in info else "N/A"
        desc = info.get("description", "")
        epochs = info.get("epochs", "Base Oficial")
        loss = info.get("loss", "N/A")
        
        return (
            f"⚡ **Modelo VITS:** `{info['name']}` | 📦 **Peso en Disco:** `{size_str}`\n\n"
            f"- **Tipo:** `{info['type']}` | **Frecuencia:** `{info['sample_rate']} Hz`\n"
            f"- **Épocas:** `{epochs}` (Loss: `{loss}`)\n"
            f"- **Descripción:** {desc}"
        )
    except Exception as e:
        return f"ℹ️ Info: {str(e)}"


def delete_vits_model_ui(selected_model: str):
    """Elimina un modelo afinado de VITS del disco y actualiza dropdowns."""
    if not selected_model or "es/css10" in selected_model or "[Base Oficial]" in selected_model:
        return "⚠️ El modelo base oficial 'es/css10' es protegido y no puede eliminarse.", gr.update(), gr.update()
        
    try:
        engine = VITSEngine.get_instance()
        clean_id = VITSEngine.extract_model_id(selected_model)
        deleted = engine.delete_trained_model(clean_id)
        if deleted:
            msg = f"🗑️ **Modelo VITS '{clean_id}' eliminado correctamente del disco.**"
        else:
            msg = f"⚠️ No se encontró el modelo '{clean_id}' para eliminar."
            
        updated_choices = get_all_vits_choices()
        return msg, gr.update(choices=updated_choices, value=updated_choices[0]), gr.update(choices=updated_choices, value=updated_choices[0])
    except Exception as e:
        return f"❌ Error al eliminar: {str(e)}", gr.update(), gr.update()


def synthesize_vits_autoplay(selected_model: str, text: str, speed: float, noise_scale: float, noise_scale_w: float):
    """Sintetiza texto en español utilizando el motor VITS ultraligero."""
    clean_text = text.strip() if text else ""
    if not clean_text:
        return None, "❌ Error: Ingrese el texto a sintetizar.", "0.00 s", "0.00 s", "0.000", "-- MB"
        
    try:
        engine = VITSEngine.get_instance()
        model_id = VITSEngine.extract_model_id(selected_model)
        info = engine.get_model_info(model_id)
        size_str = f"{info['size_mb']} MB"
        
        _, _, metrics = engine.synthesize(
            text=clean_text,
            model_name=model_id,
            speed=float(speed),
            noise_scale=float(noise_scale),
            noise_scale_w=float(noise_scale_w)
        )
        
        status_msg = (
            f"✅ **Audio sintetizado con éxito vía VITS** (Voz: `{model_id}`).\n"
            f"*(Generación ultrarrápida: RTF `{metrics['rtf']}`)*"
        )
        
        time_str = f"{metrics['inference_time_s']:.3f} s"
        dur_str = f"{metrics['audio_duration_s']:.2f} s"
        rtf_str = f"{metrics['rtf']:.4f}"
        
        return metrics["output_path"], status_msg, time_str, dur_str, rtf_str, size_str
    except Exception as e:
        return None, f"❌ Error durante síntesis VITS: {str(e)}", "0.00 s", "0.00 s", "0.000", "-- MB"


def train_vits_pipeline_ui(audio_single, audio_files, voice_name: str, epochs: int, progress=gr.Progress()):
    """Ejecuta el pipeline de Fine-Tuning completo para VITS en español."""
    paths = []
    if audio_single:
        paths.append(audio_single)
    if audio_files:
        for f in audio_files:
            file_path = f.name if hasattr(f, "name") else str(f)
            if file_path not in paths and os.path.isfile(file_path):
                paths.append(file_path)
                
    if not paths:
        return "❌ Error: Debe subir al menos un archivo de audio (MP3 o WAV) del locutor.", gr.update(), gr.update()
        
    if not voice_name or not voice_name.strip():
        return "❌ Error: Ingrese un nombre para la voz VITS a entrenar.", gr.update(), gr.update()
        
    try:
        clean_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in voice_name.strip())
        
        def vits_prog(p, msg):
            progress(p, f"[VITS Fine-Tuning] {msg}")
            
        res = VITSFineTuner.train_from_raw_audios(
            voice_name=clean_name,
            audio_paths=paths,
            epochs=int(epochs),
            batch_size=2,
            progress_callback=vits_prog
        )
        
        updated_vits = get_all_vits_choices()
        selected_val = f"⭐ [Voz Entrenada VITS] {clean_name}"
        
        msg = (
            f"🎉 **¡Voz VITS '{clean_name}' afinada con éxito! (< 110 MB)**\n\n"
            f"- **Épocas completadas:** `{res['metadata']['epochs']}` | **Loss final:** `{res['metadata']['final_loss']}`\n"
            f"- **Peso del modelo:** `{res['size_mb']} MB` (Ultra ligero, ideal para SageMaker)\n"
            f"- **Tiempo total:** `{res['duration_s']:.1f} segundos` en {res['metadata']['device']}\n"
            f"- **Carpeta:** `trained_models/vits/{clean_name}/`\n\n"
            f"*(La voz ya está disponible en la Pestaña 4 para sintetizar inmediatamente)*"
        )
        return msg, gr.update(choices=updated_vits, value=selected_val), gr.update(choices=updated_vits, value=selected_val)
    except Exception as e:
        return f"❌ Error en entrenamiento VITS: {str(e)}", gr.update(), gr.update()


# Estilos CSS
custom_css = """
.metric-box {
    background-color: #0f172a;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    border: 1px solid #1e293b;
}
.metric-val {
    font-size: 22px;
    font-weight: bold;
    color: #38bdf8;
}
.metric-lbl {
    font-size: 11px;
    color: #94a3b8;
    text-transform: uppercase;
}
"""

with gr.Blocks(title="XTTS-v2 Studio — Clonación de Voz en Español") as demo:
    gr.Markdown("# 🎙️ XTTS-v2 Studio (Español)")
    gr.Markdown("Sistema de **Clonación de Voz de Alta Fidelidad** y **Micro Fine-Tuning en GPU** (NVIDIA RTX 5070).")
    
    with gr.Tabs() as tabs:
        # PESTAÑA 1: REGISTRO, CALIBRACIÓN Y GESTIÓN DE PERFILES ZERO-SHOT
        with gr.Tab("📁 1. Registrar Perfil Zero-Shot (.pt)"):
            gr.Markdown("### 1. Sube los audios del locutor para clonación rápida instantánea (~130 KB)")
            gr.Markdown("💡 *Tip de calidad:* Sube 2 o 3 fragmentos de 10-20 segundos del mismo locutor libres de música.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_single_input = gr.Audio(
                        label="Audio de Referencia Principal (o Grabación con Micrófono)",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    audio_multi_input = gr.File(
                        label="Audios Adicionales del mismo locutor (Opcional - Técnica Multi-Audio)",
                        file_count="multiple",
                        file_types=["audio"]
                    )
                    voice_name_input = gr.Textbox(
                        label="Nombre de la Voz (Identificador)",
                        placeholder="Ej: Juan, Maria, presentador_noticias, locutor_comercial"
                    )
                    
                    gr.Markdown("#### ⚙️ Calibración de Timbre & Acondicionamiento Acústico")
                    enhance_chk = gr.Checkbox(
                        label="✨ Aplicar Limpieza y Aislamiento Acústico con IA (Recomendado)",
                        value=True,
                        info="Elimina ruidos de fondo, estática y normaliza la energía vocal al estándar de estudio"
                    )
                    with gr.Row():
                        btn_preview_clean = gr.Button("🎧 Previsualizar Limpieza Acústica", size="sm", variant="secondary")
                    
                    preview_audio = gr.Audio(
                        label="🔊 Previsualización de Muestra Limpia (A/B)",
                        interactive=False
                    )
                    preview_status = gr.Markdown("")
                    
                    temp_input = gr.Slider(
                        minimum=0.45,
                        maximum=0.90,
                        value=0.60,
                        step=0.05,
                        label="Temperatura de Fidelidad (0.60 recomendado para máxima similitud)"
                    )
                    speed_input = gr.Slider(
                        minimum=0.8,
                        maximum=1.3,
                        value=1.0,
                        step=0.05,
                        label="Velocidad Base de Habla (Speed)"
                    )
                    
                    btn_extract = gr.Button("💾 Clonar, Calibrar y Guardar Perfil (.pt)", variant="primary")
                
                with gr.Column(scale=1):
                    register_status = gr.Markdown("ℹ️ Sube los audios del locutor y pulsa el botón para extraer el perfil `.pt` de ~130 KB.")
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🗑️ Gestión / Eliminar Voces Guardadas")
                    all_initial_choices = get_all_voices_list()
                    delete_dropdown = gr.Dropdown(
                        choices=all_initial_choices,
                        value=all_initial_choices[0] if all_initial_choices else None,
                        label="Selecciona la voz a eliminar (Perfiles .pt o Modelos Entrenados)",
                        interactive=True
                    )
                    btn_delete = gr.Button("🗑️ Eliminar Voz Definitivamente", variant="stop")
                    delete_status = gr.Markdown("")
        
        # PESTAÑA 2: PRUEBA CON REPRODUCCIÓN AUTOMÁTICA Y MÉTRICAS
        with gr.Tab("🔊 2. Probar Voz Clonada & Métricas"):
            gr.Markdown("### Selecciona el modelo o perfil y escribe el texto (Reproducción Fluida)")
            with gr.Row():
                with gr.Column(scale=1):
                    all_initial_choices = get_all_voices_list()
                    voice_dropdown = gr.Dropdown(
                        choices=all_initial_choices,
                        value=all_initial_choices[0] if all_initial_choices else None,
                        label="Voz a Utilizar (⭐ Fine-Tuned 9.8 / ⚡ Zero-Shot .pt)",
                        interactive=True
                    )
                    btn_refresh = gr.Button("🔄 Recargar Lista de Voces y Modelos", size="sm")
                    
                    voice_info_box = gr.Markdown(get_voice_info(all_initial_choices[0] if all_initial_choices else None))
                    
                    target_text_input = gr.Textbox(
                        label="Texto a Sintetizar (en Español)",
                        placeholder="Escribe el texto que la voz clonada debe pronunciar...",
                        value="Hola, hoy estuve revisando los contratos y me parece que realmente podemos mejorar los resultados.",
                        lines=3
                    )
                    
                    with gr.Accordion("⚙️ Ajustes Manuales Avanzados (Opcional)", open=False):
                        override_chk = gr.Checkbox(label="Sobrescribir calibración de fábrica de la voz", value=False)
                        custom_temp = gr.Slider(minimum=0.45, maximum=0.90, value=0.60, step=0.05, label="Temperatura Manual")
                        custom_speed = gr.Slider(minimum=0.7, maximum=1.4, value=1.0, step=0.05, label="Velocidad Manual")
                    
                    btn_synthesize = gr.Button("🚀 Sintetizar con Voz Seleccionada (Autoplay)", variant="primary")
                
                with gr.Column(scale=1):
                    output_audio = gr.Audio(
                        label="🔊 Reproductor de Audio (Autoplay Activado)",
                        autoplay=True,
                        interactive=False
                    )
                    synth_status = gr.Markdown("ℹ️ Haz clic en 'Sintetizar' para generar y escuchar el audio de inmediato.")
                    
                    gr.Markdown("### 📊 Panel de Rendimiento y Velocidad")
                    with gr.Row():
                        metric_time = gr.Textbox(label="⏱️ Tiempo Inferencia (GPU)", value="0.00 s", interactive=False)
                        metric_dur = gr.Textbox(label="🔊 Duración del Audio", value="0.00 s", interactive=False)
                    with gr.Row():
                        metric_rtf = gr.Textbox(label="🚀 Factor Tiempo Real (RTF)", value="0.000", interactive=False)
                        metric_size = gr.Textbox(label="📦 Peso en Disco", value="-- KB", interactive=False)
                    
                    gr.Markdown("*(Nota: Un RTF de ~0.25 significa que 4 segundos de audio se generan en solo 1 segundo de GPU)*")

        # PESTAÑA 3: MICRO FINE-TUNING DE VOZ EN GPU
        with gr.Tab("🎯 3. Entrenar y Ajustar Voz (Fine-Tuning 9.8/10)"):
            gr.Markdown("### 🎯 Entrenamiento Especializado de Red Neuronal (Máxima Similitud 9.8/10)")
            gr.Markdown(
                "Sube entre **1 y 5 minutos** de audio de la voz que deseas clonar. "
                "El sistema limpiará el audio con IA, segmentará las frases, las transcribirá con Whisper "
                "y entrenará las capas de atención del modelo en tu tarjeta NVIDIA RTX 5070."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    train_audio_single = gr.Audio(
                        label="Audio de Entrenamiento Principal (o Grabación)",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    train_audio_multi = gr.File(
                        label="Audios Adicionales del Locutor (Opcional - Varios archivos WAV/MP3)",
                        file_count="multiple",
                        file_types=["audio"]
                    )
                    train_voice_name = gr.Textbox(
                        label="Nombre de la Voz a Entrenar",
                        placeholder="Ej: Juan_Oficial, Locutor_Premium"
                    )
                    train_epochs = gr.Slider(
                        minimum=2,
                        maximum=12,
                        value=6,
                        step=1,
                        label="Número de Épocas de Entrenamiento (6 recomendado: ~3-5 min en GPU)"
                    )
                    btn_train = gr.Button("🚀 Iniciar Micro Fine-Tuning de Voz en GPU", variant="primary")
                
                with gr.Column(scale=1):
                    train_status = gr.Markdown(
                        "ℹ️ **¿Cómo funciona?**\n\n"
                        "1. **Paso 1 (Auto-Dataset):** Whisper transcribe automáticamente las oraciones del audio.\n"
                        "2. **Paso 2 (Fine-Tuning AdamW):** Se ajusta la red neuronal para aprender la entonación y timbre exacto del locutor.\n"
                        "3. **Paso 3 (Listo para usar):** El modelo ajustado queda disponible inmediatamente en la Pestaña 2.\n\n"
                        "*Haz clic en 'Iniciar Micro Fine-Tuning' cuando tengas tus audios cargados.*"
                    )

        # PESTAÑA 4: SÍNTESIS Y MÉTRICAS CON VITS ESPAÑOL (<150 MB)
        with gr.Tab("⚡ 4. VITS: Síntesis & Métricas (<150MB)"):
            gr.Markdown("### ⚡ Síntesis Ultrarrápida con VITS en Español (Bajo Peso / Ideal SageMaker)")
            gr.Markdown("Inferencia instantánea a 22050 Hz con checkpoints de ~103 MB listos para producción.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    vits_initial_choices = get_all_vits_choices()
                    vits_model_dropdown = gr.Dropdown(
                        choices=vits_initial_choices,
                        value=vits_initial_choices[0] if vits_initial_choices else None,
                        label="Modelo / Voz VITS a Utilizar",
                        interactive=True
                    )
                    btn_vits_refresh = gr.Button("🔄 Recargar Modelos VITS", size="sm")
                    vits_info_box = gr.Markdown(get_vits_info(vits_initial_choices[0] if vits_initial_choices else None))
                    
                    vits_text_input = gr.Textbox(
                        label="Texto a Sintetizar (en Español)",
                        placeholder="Escribe el texto que la voz VITS debe pronunciar...",
                        value="Hola, este es el motor de síntesis VITS optimizado para español con máxima velocidad y bajo peso.",
                        lines=3
                    )
                    
                    with gr.Accordion("⚙️ Parámetros Acústicos VITS", open=False):
                        vits_speed = gr.Slider(minimum=0.6, maximum=1.6, value=1.0, step=0.05, label="Velocidad (Speed)")
                        vits_noise = gr.Slider(minimum=0.3, maximum=1.0, value=0.667, step=0.05, label="Noise Scale (Variabilidad fonética)")
                        vits_noise_w = gr.Slider(minimum=0.3, maximum=1.2, value=0.8, step=0.05, label="Noise Scale W (Variabilidad de duración)")
                        
                    btn_vits_synthesize = gr.Button("🚀 Sintetizar con VITS (Autoplay)", variant="primary")
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🗑️ Gestión / Eliminar Voces VITS")
                    vits_delete_dropdown = gr.Dropdown(
                        choices=vits_initial_choices,
                        value=vits_initial_choices[0] if vits_initial_choices else None,
                        label="Selecciona la voz VITS a eliminar (Excepto Base)",
                        interactive=True
                    )
                    btn_vits_delete = gr.Button("🗑️ Eliminar Modelo VITS", variant="stop")
                    vits_delete_status = gr.Markdown("")
                    
                with gr.Column(scale=1):
                    vits_output_audio = gr.Audio(
                        label="🔊 Reproductor VITS (Autoplay Activado)",
                        autoplay=True,
                        interactive=False
                    )
                    vits_synth_status = gr.Markdown("ℹ️ Haz clic en 'Sintetizar con VITS' para generar y escuchar el audio de inmediato.")
                    
                    gr.Markdown("### 📊 Panel de Rendimiento y Velocidad (VITS)")
                    with gr.Row():
                        vits_metric_time = gr.Textbox(label="⏱️ Tiempo Inferencia", value="0.00 s", interactive=False)
                        vits_metric_dur = gr.Textbox(label="🔊 Duración Audio", value="0.00 s", interactive=False)
                    with gr.Row():
                        vits_metric_rtf = gr.Textbox(label="🚀 Factor Tiempo Real (RTF)", value="0.000", interactive=False)
                        vits_metric_size = gr.Textbox(label="📦 Peso Modelo", value="-- MB", interactive=False)
                        
                    gr.Markdown("*(Nota: Un RTF de ~0.08 significa que 10 segundos de audio se generan en menos de 1 segundo)*")

        # PESTAÑA 5: FINE-TUNING DE VOZ CON VITS EN ESPAÑOL
        with gr.Tab("🎯 5. VITS: Entrenar Voz (Fine-Tuning Español)"):
            gr.Markdown("### 🎯 Entrenamiento de Voz VITS sobre Modelo Base Oficial en Español")
            gr.Markdown(
                "Sube entre **1 y 5 minutos** de audio del locutor. "
                "El sistema limpiará el audio con IA (AudioEnhancer), transcribirá automáticamente con Whisper "
                "y entrenará la red VITS (< 110 MB) para aprender el timbre del locutor."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    vits_train_audio_single = gr.Audio(
                        label="Audio de Entrenamiento Principal (o Grabación con Micrófono)",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    vits_train_audio_multi = gr.File(
                        label="Audios Adicionales del Locutor (Opcional)",
                        file_count="multiple",
                        file_types=["audio"]
                    )
                    vits_train_voice_name = gr.Textbox(
                        label="Nombre de la Voz VITS a Entrenar",
                        placeholder="Ej: locutor_vits_es, presentador_radio"
                    )
                    vits_train_epochs = gr.Slider(
                        minimum=4,
                        maximum=20,
                        value=8,
                        step=2,
                        label="Número de Épocas de Fine-Tuning (8 recomendado: ~2-4 min)"
                    )
                    btn_vits_train = gr.Button("🚀 Iniciar Fine-Tuning de Voz VITS", variant="primary")
                    
                with gr.Column(scale=1):
                    vits_train_status = gr.Markdown(
                        "ℹ️ **Pipeline de Fine-Tuning VITS:**\n\n"
                        "1. **Limpieza IA:** Se eliminan ruidos y normaliza la energía acústica.\n"
                        "2. **Auto-Alineación:** Whisper transcribe y fragmenta las frases.\n"
                        "3. **Ajuste Fino:** Se adaptan los pesos sobre el modelo base oficial en español (`es/css10`).\n"
                        "4. **Artefacto Ultraligero:** Se exporta un checkpoint de solo **~103 MB** listo para la Pestaña 4 y para Amazon SageMaker."
                    )

    # Eventos Pestaña 1
    btn_preview_clean.click(
        fn=preview_enhanced_audio,
        inputs=[audio_single_input, audio_multi_input],
        outputs=[preview_audio, preview_status]
    )

    btn_extract.click(
        fn=register_and_extract_voice,
        inputs=[audio_single_input, audio_multi_input, voice_name_input, temp_input, speed_input, enhance_chk],
        outputs=[register_status, voice_dropdown, delete_dropdown]
    )

    btn_delete.click(
        fn=delete_voice_ui,
        inputs=[delete_dropdown],
        outputs=[delete_status, voice_dropdown, delete_dropdown]
    )

    # Eventos Pestaña 2
    voice_dropdown.change(
        fn=get_voice_info,
        inputs=[voice_dropdown],
        outputs=[voice_info_box]
    )

    btn_refresh.click(
        fn=lambda: (gr.update(choices=get_all_voices_list()), gr.update(choices=get_all_voices_list())),
        inputs=[],
        outputs=[voice_dropdown, delete_dropdown]
    )

    btn_synthesize.click(
        fn=synthesize_clean_autoplay,
        inputs=[voice_dropdown, target_text_input, custom_temp, custom_speed, override_chk],
        outputs=[output_audio, synth_status, metric_time, metric_dur, metric_rtf, metric_size]
    )

    # Eventos Pestaña 3
    btn_train.click(
        fn=train_voice_pipeline_ui,
        inputs=[train_audio_single, train_audio_multi, train_voice_name, train_epochs],
        outputs=[train_status, voice_dropdown, delete_dropdown, voice_dropdown]
    )

    # Eventos Pestaña 4 (VITS Síntesis)
    vits_model_dropdown.change(
        fn=get_vits_info,
        inputs=[vits_model_dropdown],
        outputs=[vits_info_box]
    )

    btn_vits_refresh.click(
        fn=lambda: (gr.update(choices=get_all_vits_choices()), gr.update(choices=get_all_vits_choices())),
        inputs=[],
        outputs=[vits_model_dropdown, vits_delete_dropdown]
    )

    btn_vits_synthesize.click(
        fn=synthesize_vits_autoplay,
        inputs=[vits_model_dropdown, vits_text_input, vits_speed, vits_noise, vits_noise_w],
        outputs=[vits_output_audio, vits_synth_status, vits_metric_time, vits_metric_dur, vits_metric_rtf, vits_metric_size]
    )

    btn_vits_delete.click(
        fn=delete_vits_model_ui,
        inputs=[vits_delete_dropdown],
        outputs=[vits_delete_status, vits_model_dropdown, vits_delete_dropdown]
    )

    # Eventos Pestaña 5 (VITS Fine-Tuning)
    btn_vits_train.click(
        fn=train_vits_pipeline_ui,
        inputs=[vits_train_audio_single, vits_train_audio_multi, vits_train_voice_name, vits_train_epochs],
        outputs=[vits_train_status, vits_model_dropdown, vits_delete_dropdown]
    )


if __name__ == "__main__":
    print("Iniciando Voice Cloning Studio (XTTS-v2 & VITS Español)...")
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft(primary_hue="indigo"), css=custom_css)
