# 🎙️ Voice Cloning Studio — XTTS-v2 & VITS (Español)

Plataforma integral de clonación, síntesis y entrenamiento (*fine-tuning*) de voces en español con aceleración por hardware (GPU CUDA) o CPU, diseñada con arquitectura desacoplada para despliegue local o integración en la nube (Amazon SageMaker / microservicios).

---

## 🌟 Características Principales

1. **XTTS-v2 (Zero-Shot & Few-Shot Multilingüe/Español):**
   - Extracción de **perfiles de voz ultraligeros (`.pt` de ~130 KB)** con tensores condicionantes (`gpt_cond_latent` y `speaker_embedding`), eliminando la necesidad de reprocesar audios pesados en cada síntesis.
   - Síntesis natural a **24 kHz** con alta fidelidad tímbrica y entonación humana.
   - Soporte para **Fine-Tuning de pesos** para maximizar la identidad vocal del locutor.

2. **VITS en Español (CSS10):**
   - Motor generativo *end-to-end* ultraligero (**~103 MB de checkpoint**).
   - Inferencia ultrarrápida con factor de tiempo real en GPU (**RTF < 0.15**).
   - Módulo de micro *fine-tuning* transferible y desacoplado de XTTS.

3. **AudioEnhancer (Acondicionamiento Acústico Inteligente):**
   - Filtro Paso-Banda Butterworth (75 Hz - 11.5 kHz) para eliminar *rumble*, frecuencias parásitas y siseos.
   - Reducción adaptativa de ruido espectral (*Wiener / STFT Spectral Gating*).
   - Recorte inteligente de silencios muertos (VAD) y normalización RMS estándar broadcast a **-16 dBFS** con limitador *anti-clipping*.

4. **DatasetPreparer (Pipeline Automático de Entrenamiento):**
   - Segmentación automática de audios largos en clips óptimos de 2 a 10 segundos.
   - Transcripción fonética y textual automática utilizando **OpenAI Whisper**.
   - Generación automática de `metadata.csv` normalizado.

5. **Interfaz Web Reactiva (Gradio):**
   - Estudio visual en 5 pestañas con reproducción automática (*autoplay*), gestión de perfiles/modelos y telemetría de rendimiento en tiempo real (RTF, duración, tiempo de inferencia y tamaño).

---

## 📋 Requisitos del Sistema

- **Sistema Operativo:** Windows 10/11 o Linux (Ubuntu 20.04+).
- **Python:** Versión **3.10**, **3.11** o **3.12** (probado en Python 3.12.10).
- **GPU (Recomendado):** Tarjeta NVIDIA con soporte CUDA (ej. RTX 30xx, 40xx, 50xx con 6 GB+ VRAM). Compatible con fallback en CPU.
- **Herramientas de Sistema:**
  - [FFmpeg](https://ffmpeg.org/): Necesario para la decodificación y conversión de múltiples formatos de audio (MP3, M4A, WAV, etc.).

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd trabajo-voces-clonadas
```

### 2. Crear y activar el entorno virtual

**Opción recomendada (usando `uv` de Astral - ultra rápido):**
```bash
# Instalar uv si no lo tienes: pip install uv
uv venv .venv --python 3.12
```

**Opción estándar (`venv` de Python):**
```bash
python -m venv .venv
```

**Activar el entorno:**
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

---

### 3. Instalar PyTorch con aceleración CUDA

Instala la versión de PyTorch adecuada para tu tarjeta gráfica antes de instalar las dependencias:

- **Para GPUs NVIDIA modernas (CUDA 12.x / 12.8 / RTX 50 Series):**
  ```bash
  pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  ```
- **Para CUDA 12.4 / 12.1 (RTX 30xx / 40xx):**
  ```bash
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```
- **Para ejecutar exclusivamente en CPU:**
  ```bash
  pip install torch torchaudio
  ```

---

### 4. Instalar las dependencias del proyecto

Con el entorno virtual activado:
```bash
pip install -r requirements.txt
```
*(o si usas `uv`: `uv pip install -r requirements.txt`)*

---

### 5. Descargar los Pesos del Modelo XTTS-v2

El motor XTTS-v2 espera los pesos base dentro del directorio `xtts_models/`. Si es la primera vez que levantas el repositorio, puedes descargarlos automáticamente con este comando de una sola línea en Python:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='coqui/XTTS-v2', local_dir='xtts_models', allow_patterns=['config.json', 'vocab.json', '*.pth', 'LICENSE.txt'])"
```

Los archivos requeridos en `xtts_models/` son:
- `model.pth` (~1.8 GB)
- `config.json`
- `vocab.json`
- `dvae.pth`
- `mel_stats.pth`
- `speakers_xtts.pth`

> ℹ️ **Nota:** El modelo base de VITS (`tts_models/es/css10/vits`) y el modelo Whisper (`openai/whisper-tiny`) se descargan automáticamente en la caché local en su primera ejecución.

---

### 6. Instalar FFmpeg en el sistema

Asegúrate de que `ffmpeg` esté en las variables de entorno (`PATH`):
- **Windows (Winget):**
  ```powershell
  winget install Gyan.FFmpeg
  ```
  *(Luego reinicia la terminal)*
- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg
  ```

---

## 🖥️ Cómo Levantar el Sistema

### Modo A: Interfaz Web Interactiva (Gradio)

Para iniciar el estudio visual completo:
```bash
python app_xtts.py
```

Una vez cargado, abre tu navegador en:
👉 **`http://127.0.0.1:7860`**

#### Pestañas de la Interfaz Web:
1. **Pestaña 1 — Calibrar y Extraer Perfil (.pt / XTTS-v2):**
   - Sube uno o varios audios de referencia (MP3 o WAV, de 10 a 60 segundos recomendados).
   - Activa el **Filtro Acústico Inteligente (AudioEnhancer)** para limpiar ruidos de fondo, eco y siseos.
   - Escucha la previsualización limpia y haz clic en **"Extraer y Guardar Perfil (.pt)"**.
2. **Pestaña 2 — Síntesis de Voz XTTS-v2:**
   - Selecciona una voz registrada (perfil `.pt` o modelo fine-tuned).
   - Escribe el texto en español a sintetizar.
   - Ajusta velocidad y temperatura (recomendado: 0.65).
   - Generación con reproducción inmediata y métricas de RTF en pantalla.
3. **Pestaña 3 — Fine-Tuning de Voz XTTS-v2:**
   - Sube audios de 1 a 5 minutos del locutor.
   - Ajusta las épocas (ej. 6 - 8 épocas).
   - El sistema procesa los audios con Whisper, entrena y genera un checkpoint afinado en `trained_models/`.
4. **Pestaña 4 — Síntesis VITS en Español:**
   - Síntesis ultrarrápida a 22.05 kHz con el checkpoint base o tus modelos VITS afinados.
   - Control de velocidad (*speed*), ruido de fonemas (*noise_scale*) y variabilidad (*noise_scale_w*).
5. **Pestaña 5 — Fine-Tuning de Voz VITS:**
   - Entrenamiento ligero para adaptar el modelo base en español (`es/css10`) al locutor en 2 a 4 minutos.

---

### Modo B: Síntesis desde Línea de Comandos (CLI)

Puedes sintetizar audios directamente sin levantar la interfaz web mediante `clone_xtts_es.py`:

```bash
# Sintetizar usando un perfil de voz registrado (.pt)
python clone_xtts_es.py --voice juan --texto "Hola, esta es una prueba de voz clonada con XTTS-v2." --out salida.wav

# Sintetizar pasando un audio de referencia directo
python clone_xtts_es.py --voice "reference_audios/muestra_maria.wav" --texto "Bienvenido al sistema automatizado de voz." --out bienvenida.wav

# Sintetizar leyendo el texto desde un archivo .txt
python clone_xtts_es.py --voice jose --texto script.txt --out resultado.wav --speed 1.05 --temperature 0.65
```

**Parámetros CLI:**
- `--voice`: Nombre del perfil guardado en `voices/` o ruta a archivo de audio `.mp3`/`.wav`/`.pt`.
- `--texto`: Texto inline o ruta a un archivo `.txt`.
- `--out`: Ruta del archivo `.wav` resultante (default: `salida_xtts.wav`).
- `--speed`: Velocidad de locución (default: `1.0`).
- `--temperature`: Temperatura / variabilidad del modelo (default: `0.65`).

---

## 🧪 Validación y Pruebas Automatizadas

El proyecto cuenta con suites de pruebas automatizadas:

```bash
# Validar motor y flujo XTTS-v2
python test_xtts_es.py

# Validar motor y flujo VITS
python test_vits_es.py

# Validar pipeline de AudioEnhancer y acondicionamiento
python test_enhancer_and_cloning.py

# Validar pipeline de segmentación Whisper y Fine-Tuning
python test_finetuning_pipeline.py
```

---

## 📁 Estructura del Proyecto

```text
trabajo-voces-clonadas/
├── app_xtts.py                  # Interfaz Web interactiva Gradio (5 pestañas)
├── clone_xtts_es.py             # CLI para síntesis directa por terminal
├── xtts_engine.py               # Motor central XTTS-v2 (extracción .pt e inferencia)
├── vits_engine.py               # Motor central VITS (inferencia desacoplada)
├── audio_enhancer.py            # Pipeline de acondicionamiento y limpieza acústica
├── dataset_preparer.py          # Segmentación VAD y transcripción automática con Whisper
├── xtts_finetuner.py            # Pipeline de Fine-Tuning para XTTS-v2
├── vits_finetuner.py            # Pipeline de Fine-Tuning para VITS
├── requirements.txt             # Dependencias del proyecto
├── .gitignore                   # Exclusiones de Git (specs, modelos, cachés, etc.)
│
├── xtts_models/                 # Pesos descargados de Coqui XTTS-v2
│   ├── model.pth
│   ├── config.json
│   ├── vocab.json
│   └── ...
├── voices/                      # Perfiles de voz latentes (.pt de ~130 KB)
│   └── *.pt
├── trained_models/              # Checkpoints generados por Fine-Tuning
│   └── vits/
├── datasets/                    # Datasets temporales/preparados por Whisper
└── outputs/                     # Audios sintetizados generados por la UI o tests
```

---

## 🛠️ Preguntas Frecuentes y Solución de Problemas

### 1. `CUDA out of memory` (Falta de memoria VRAM)
- Cierra otras aplicaciones que consuman GPU (navegadores con aceleración pesada, juegos o editores de video).
- Reduce la longitud del texto a sintetizar en bloques o utiliza el modo streaming.
- Si no cuentas con GPU suficiente, el sistema automáticamente o explícitamente puede correr en CPU (`device="cpu"`).

### 2. Error al importar `soundfile` o decodificar audio
- Asegúrate de haber instalado `ffmpeg` y haber reiniciado la consola para que el `PATH` se actualice.

### 3. Caracteres extraños o errores de encoding en Windows
- Los scripts incorporan reconfiguración UTF-8 automática en Windows (`sys.stdout.reconfigure(encoding="utf-8")`), pero se recomienda usar **Windows Terminal** o PowerShell con codificación UTF-8 habilitada.

---

## 📄 Licencia

Este proyecto integra modelos de código abierto desarrollados bajo la iniciativa de Coqui TTS (XTTS-v2 bajo licencia Coqui Public Model License / CPML, y VITS bajo Apache 2.0 / LGPL). Consulte los términos de licencia correspondientes para usos comerciales.
