#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clone_xtts_es.py
----------------
Wrapper CLI para clonación y síntesis de voz en español usando XTTS-v2.
Permite pasar un perfil .pt guardado, un nombre de voz registrada o un audio de referencia directo.
"""

import argparse
import io
import os
import sys

# Forzar UTF-8 en Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xtts_engine import XTTSEngine


def resolve_text_input(texto_arg: str) -> str:
    """Resuelve si el argumento es texto inline o una ruta a un archivo .txt."""
    if os.path.isfile(texto_arg):
        with open(texto_arg, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"El archivo de texto '{texto_arg}' está vacío.")
            return content
    
    texto_limpio = texto_arg.strip()
    if not texto_limpio:
        raise ValueError("El texto a sintetizar no puede estar vacío.")
    return texto_limpio


def main():
    parser = argparse.ArgumentParser(description="Clonación de voz zero-shot en español con XTTS-v2.")
    parser.add_argument("--voice", required=True, help="Nombre de la voz guardada, archivo .pt o archivo de audio (.mp3/.wav)")
    parser.add_argument("--texto", required=True, help="Texto a sintetizar en español o ruta a archivo .txt")
    parser.add_argument("--out", required=False, default="salida_xtts.wav", help="Ruta de salida del archivo .wav (default: salida_xtts.wav)")
    parser.add_argument("--speed", required=False, type=float, default=1.0, help="Velocidad del habla (default: 1.0)")
    parser.add_argument("--temperature", required=False, type=float, default=0.65, help="Temperatura / Fidelidad (default: 0.65)")
    
    args = parser.parse_args()

    try:
        texto = resolve_text_input(args.texto)
        engine = XTTSEngine.get_instance()
        
        print(f"\n[XTTS-v2] Sintetizando con voz: '{args.voice}' (Temp: {args.temperature:.2f})...")
        res = engine.synthesize(
            text=texto,
            voice_identifier=args.voice,
            output_path=args.out,
            language="es",
            speed=args.speed,
            temperature=args.temperature
        )

        print("==========================================")
        print(" OK: Audio generado exitosamente con XTTS-v2")
        print(f" Ruta de salida       : {res['output_path']}")
        print(f" Duración del audio   : {res['duration_s']:.2f} s")
        print(f" Tiempo de inferencia : {res['infer_time_s']:.2f} s")
        print(f" Factor Tiempo Real   : {res['rtf']:.3f} (RTF)")
        print("==========================================")
        sys.exit(0)

    except Exception as e:
        sys.stderr.write(f"\nERROR: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
