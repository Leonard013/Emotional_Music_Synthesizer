"""Emotional Music Synthesizer — main entry point.

Usage:
    # Full pipeline: record voice, detect emotion, generate music
    python main.py

    # Skip recording, provide pre-existing files
    python main.py --voice_midi voice_basic_pitch.mid --photo photo.jpg

    # Specify emotion directly (skip camera)
    python main.py --voice_midi voice_basic_pitch.mid --emotion happy

    # Adjust generation parameters
    python main.py --voice_midi voice_basic_pitch.mid --emotion sad --num_notes 200 --temperature 1.5
"""

import argparse
import subprocess
import sys

import numpy as np
import tensorflow as tf

from config import (
    SEED,
    RECORD_SECONDS,
    VOICE_WAV_PATH,
    VOICE_MIDI_PATH,
    PHOTO_PATH,
    OUTPUT_MIDI_PATH,
    TEMPERATURE,
    NUM_PREDICTIONS,
)


def record_voice(output_wav: str, seconds: int = RECORD_SECONDS):
    """Record audio from the microphone using sounddevice and scipy."""
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write

        print(f"Recording {seconds} seconds of audio...")
        fs = 44100
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        write(output_wav, fs, recording)
        print(f"Saved to {output_wav}")
    except ImportError:
        print("sounddevice not installed. Install with: pip install sounddevice")
        print("Alternatively, provide a pre-recorded WAV file.")
        sys.exit(1)


def convert_voice_to_midi(wav_path: str, midi_path: str):
    """Convert a WAV file to MIDI using basic-pitch."""
    import os

    output_dir = os.path.dirname(midi_path) or "."
    print(f"Converting {wav_path} to MIDI...")
    subprocess.run(
        ["basic-pitch", output_dir, wav_path],
        check=True,
    )
    # basic-pitch names the output based on the input filename
    base = os.path.splitext(os.path.basename(wav_path))[0]
    generated = os.path.join(output_dir, f"{base}_basic_pitch.mid")
    if generated != midi_path and os.path.exists(generated):
        os.rename(generated, midi_path)
    print(f"MIDI saved to {midi_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate emotion-responsive piano music"
    )
    parser.add_argument(
        "--voice_midi", type=str, default=None,
        help="Path to pre-existing voice MIDI file (skip recording)"
    )
    parser.add_argument(
        "--voice_wav", type=str, default=None,
        help="Path to pre-existing voice WAV file (skip recording, still converts to MIDI)"
    )
    parser.add_argument(
        "--photo", type=str, default=None,
        help="Path to pre-existing face photo (skip webcam capture)"
    )
    parser.add_argument(
        "--emotion", type=str, choices=["happy", "sad"], default=None,
        help="Specify emotion directly instead of detecting from photo"
    )
    parser.add_argument(
        "--num_notes", type=int, default=NUM_PREDICTIONS,
        help=f"Number of notes to generate (default: {NUM_PREDICTIONS})"
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_MIDI_PATH,
        help=f"Output MIDI file path (default: {OUTPUT_MIDI_PATH})"
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="Disable piano roll plot"
    )
    args = parser.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Step 1: Get voice MIDI
    if args.voice_midi:
        voice_midi = args.voice_midi
        print(f"Using existing voice MIDI: {voice_midi}")
    elif args.voice_wav:
        voice_midi = VOICE_MIDI_PATH
        convert_voice_to_midi(args.voice_wav, voice_midi)
    else:
        record_voice(VOICE_WAV_PATH)
        voice_midi = VOICE_MIDI_PATH
        convert_voice_to_midi(VOICE_WAV_PATH, voice_midi)

    # Step 2: Get emotion label
    if args.emotion:
        label = 1 if args.emotion == "happy" else -1
        print(f"Using specified emotion: {args.emotion} (label={label})")
    elif args.photo:
        from emotion_detection import detect_emotion_from_image

        label = detect_emotion_from_image(args.photo)
        emotion_str = "happy" if label == 1 else "sad"
        print(f"Detected emotion: {emotion_str} (label={label})")
    else:
        from emotion_detection import detect_emotion_from_webcam

        label = detect_emotion_from_webcam()
        emotion_str = "happy" if label == 1 else "sad"
        print(f"Detected emotion: {emotion_str} (label={label})")

    # Step 3: Generate music
    from generate import generate_music

    generated_notes = generate_music(
        voice_midi_path=voice_midi,
        label=label,
        output_path=args.output,
        num_predictions=args.num_notes,
        temperature=args.temperature,
        show_plot=not args.no_plot,
    )

    print(f"\nGenerated {len(generated_notes)} notes -> {args.output}")


if __name__ == "__main__":
    main()
