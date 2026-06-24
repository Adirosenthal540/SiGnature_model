"""Preprocess an audio file for SiGnature inference.

Generates:
  1. A .TextGrid file (Whisper word-level alignment)
  2. A .txt file (tokenized transcript matching the model's vocabulary)
  3. A _semantic.txt template (same tokenized text, ready for SeG gesture annotation)

Usage:
  python -m sample.preprocess_audio --audio_path ./path/to/audio.wav
  python -m sample.preprocess_audio --audio_path ./path/to/audio.wav --list_gestures
"""
import argparse
import os
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio for SiGnature inference")
    parser.add_argument("--audio_path", required=True, type=str, help="Path to a .wav audio file")
    parser.add_argument("--list_gestures", action="store_true", help="Print available SeG gesture codes from SeG_list.xlsx")
    parser.add_argument("--seg_xlsx", default="./datasets/SeG_SMPLX/SeG_list.xlsx", help="Path to SeG_list.xlsx (for --list_gestures)")
    parser.add_argument("--data_path", default="./datasets/BEAT_SMPL/BEAT2/beat_english_v2.0.0/", help="Path to BEAT2 data (for vocab)")
    args = parser.parse_args()

    assert os.path.exists(args.audio_path), f"Audio file not found: {args.audio_path}"

    out_dir = os.path.dirname(os.path.abspath(args.audio_path))
    name = os.path.splitext(os.path.basename(args.audio_path))[0]

    # Step 1: Generate TextGrid via Whisper
    textgrid_path = os.path.join(out_dir, f"{name}.TextGrid")
    from data_loaders.beat2.utils.cache_utils import AudioToTextgrid

    audio_to_tg = AudioToTextgrid()
    print(f"Transcribing {args.audio_path} with Whisper...")
    audio_to_tg.audio_to_textgrid(args.audio_path, textgrid_path)
    print(f"TextGrid saved to: {textgrid_path}")

    # Step 2: Build tokenized transcript from TextGrid (matches what the model sees)
    import textgrid as tg

    from data_loaders.beat2.beat2_dataset import _load_vocab
    with open(os.path.join(args.data_path, "weights/vocab.pkl"), "rb") as f:
        lang_model = _load_vocab(f)

    tgrid = tg.TextGrid.fromFile(textgrid_path)
    words = []
    for word in tgrid[0]:
        word_n = word.mark
        if word_n and word_n.strip():
            words.append(word_n)
    tokenized_text = " ".join(words)

    # Step 3: Save plain transcript
    txt_path = os.path.join(out_dir, f"{name}.txt")
    with open(txt_path, "w") as f:
        f.write(tokenized_text)
    print(f"Transcript saved to: {txt_path}")

    # Step 4: Create semantic annotation template
    semantic_path = os.path.join(out_dir, f"{name}_semantic.txt")
    with open(semantic_path, "w") as f:
        f.write(tokenized_text)
    print(f"Semantic template saved to: {semantic_path}")
    print("  Edit this file to add gesture codes inline, e.g.:")
    print('  "the first thing (1231 FOREFINGER RAISE-ONE) i like to do..."')

    # Step 4: Optionally list available gestures
    if args.list_gestures:
        print_available_gestures(args.seg_xlsx)


def print_available_gestures(xlsx_path):
    import pandas as pd

    if not os.path.exists(xlsx_path):
        print(f"\nWarning: SeG xlsx not found at {xlsx_path}")
        print("  Download it with: bash prepare/download_SeG.sh")
        return

    df = pd.read_excel(xlsx_path, header=2)
    print("\nAvailable SeG Gesture Codes:")
    print("-" * 60)
    for _, row in df.iterrows():
        sem_idx = row.get("Semantics-Aware Index", "?")
        label = row.get("Label", "?")
        desc = row.get("Description", "")
        if pd.notna(sem_idx) and pd.notna(label):
            print(f"  ({int(sem_idx) if isinstance(sem_idx, float) else sem_idx} {label}): {desc}")


if __name__ == "__main__":
    main()
