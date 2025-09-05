"""
Real-time processing for whistle-to-music conversion.
Handles live audio input and streaming output.
"""

import torch
import numpy as np
import librosa
import sounddevice as sd
import threading
import queue
import time
from typing import Optional, Callable, Dict, Any
from collections import deque
import warnings

from ..whistle_analysis import AudioProcessor, FeatureExtractor, PitchDetector
from ..data_processing import DataPreprocessor
from .whistle_to_music import WhistleToMusicConverter


class RealTimeProcessor:
    """Real-time whistle-to-music processor."""
    
    def __init__(self, 
                 model_path: str,
                 input_device: Optional[int] = None,
                 output_device: Optional[int] = None,
                 sample_rate: int = 22050,
                 chunk_size: int = 1024,
                 buffer_size: int = 10,
                 device: str = "auto"):
        """
        Initialize real-time processor.
        
        Args:
            model_path: Path to trained model
            input_device: Audio input device ID
            output_device: Audio output device ID
            sample_rate: Audio sample rate
            chunk_size: Size of audio chunks to process
            buffer_size: Number of chunks to buffer
            device: Device for model inference
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        
        # Initialize converter
        self.converter = WhistleToMusicConverter(
            model_path=model_path,
            device=device,
            sample_rate=sample_rate
        )
        
        # Audio buffers
        self.input_buffer = deque(maxlen=buffer_size)
        self.output_buffer = queue.Queue(maxsize=buffer_size)
        
        # Processing state
        self.is_processing = False
        self.is_recording = False
        self.processing_thread = None
        self.audio_thread = None
        
        # Audio devices
        self.input_device = input_device
        self.output_device = output_device
        
        # Callbacks
        self.on_music_generated: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Processing parameters
        self.min_whistle_duration = 2.0  # Minimum duration to process
        self.silence_threshold = 0.01  # Silence detection threshold
        self.temperature = 0.8
        self.max_length = 256
    
    def start_processing(self):
        """Start real-time processing."""
        if self.is_processing:
            print("Processing already started")
            return
        
        self.is_processing = True
        self.is_recording = True
        
        # Start audio thread
        self.audio_thread = threading.Thread(target=self._audio_loop)
        self.audio_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        print("Real-time processing started")
    
    def stop_processing(self):
        """Stop real-time processing."""
        self.is_processing = False
        self.is_recording = False
        
        if self.audio_thread:
            self.audio_thread.join()
        
        if self.processing_thread:
            self.processing_thread.join()
        
        print("Real-time processing stopped")
    
    def _audio_loop(self):
        """Main audio processing loop."""
        try:
            with sd.InputStream(
                device=self.input_device,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            ):
                while self.is_recording:
                    time.sleep(0.01)
        
        except Exception as e:
            if self.on_error:
                self.on_error(f"Audio error: {e}")
            else:
                print(f"Audio error: {e}")
    
    def _audio_callback(self, indata, frames, time, status):
        """Audio input callback."""
        if status:
            print(f"Audio status: {status}")
        
        # Convert to numpy array and add to buffer
        audio_chunk = indata[:, 0]  # Take first channel
        self.input_buffer.append(audio_chunk.copy())
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_processing:
            try:
                # Check if we have enough audio data
                if len(self.input_buffer) < self.buffer_size:
                    time.sleep(0.01)
                    continue
                
                # Get audio data
                audio_data = np.concatenate(list(self.input_buffer))
                
                # Check for silence
                if self._is_silence(audio_data):
                    time.sleep(0.1)
                    continue
                
                # Check minimum duration
                duration = len(audio_data) / self.sample_rate
                if duration < self.min_whistle_duration:
                    time.sleep(0.1)
                    continue
                
                # Process whistle
                self._process_whistle(audio_data)
                
                # Clear buffer after processing
                self.input_buffer.clear()
                
            except Exception as e:
                if self.on_error:
                    self.on_error(f"Processing error: {e}")
                else:
                    print(f"Processing error: {e}")
                time.sleep(0.1)
    
    def _is_silence(self, audio_data: np.ndarray) -> bool:
        """Check if audio data is silence."""
        rms = np.sqrt(np.mean(audio_data**2))
        return rms < self.silence_threshold
    
    def _process_whistle(self, audio_data: np.ndarray):
        """Process a whistle segment."""
        try:
            # Convert whistle to music
            result = self.converter.convert_whistle_to_music(
                whistle_audio=audio_data,
                max_length=self.max_length,
                temperature=self.temperature,
                top_k=50,
                top_p=0.9
            )
            
            # Add to output buffer
            if not self.output_buffer.full():
                self.output_buffer.put(result)
            
            # Call callback if provided
            if self.on_music_generated:
                self.on_music_generated(result)
            
        except Exception as e:
            if self.on_error:
                self.on_error(f"Whistle processing error: {e}")
            else:
                print(f"Whistle processing error: {e}")
    
    def get_generated_music(self) -> Optional[Dict]:
        """Get the latest generated music."""
        try:
            return self.output_buffer.get_nowait()
        except queue.Empty:
            return None
    
    def set_processing_parameters(self, **kwargs):
        """Set processing parameters."""
        if 'temperature' in kwargs:
            self.temperature = kwargs['temperature']
        if 'max_length' in kwargs:
            self.max_length = kwargs['max_length']
        if 'min_whistle_duration' in kwargs:
            self.min_whistle_duration = kwargs['min_whistle_duration']
        if 'silence_threshold' in kwargs:
            self.silence_threshold = kwargs['silence_threshold']
    
    def set_callbacks(self, 
                     on_music_generated: Optional[Callable] = None,
                     on_error: Optional[Callable] = None):
        """Set callback functions."""
        if on_music_generated:
            self.on_music_generated = on_music_generated
        if on_error:
            self.on_error = on_error


class StreamingMusicGenerator:
    """Streaming music generator for real-time output."""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 output_device: Optional[int] = None,
                 buffer_size: int = 1024):
        """
        Initialize streaming music generator.
        
        Args:
            sample_rate: Audio sample rate
            output_device: Audio output device ID
            buffer_size: Audio buffer size
        """
        self.sample_rate = sample_rate
        self.output_device = output_device
        self.buffer_size = buffer_size
        
        # Audio output
        self.output_stream = None
        self.is_playing = False
        
        # MIDI to audio conversion
        self.synthesizer = None
        self._setup_synthesizer()
    
    def _setup_synthesizer(self):
        """Setup MIDI synthesizer."""
        try:
            import fluidsynth
            self.synthesizer = fluidsynth.Synth()
            self.synthesizer.start()
            
            # Load a soundfont
            sfid = self.synthesizer.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")
            self.synthesizer.program_select(0, sfid, 0, 0)  # Piano
            
        except ImportError:
            print("FluidSynth not available, using simple sine wave synthesis")
            self.synthesizer = None
    
    def play_midi(self, midi_data: Dict):
        """Play MIDI data as audio."""
        if not self.is_playing:
            self.start_playback()
        
        # Convert MIDI to audio
        audio_data = self._midi_to_audio(midi_data)
        
        # Play audio
        if audio_data is not None:
            self._play_audio(audio_data)
    
    def _midi_to_audio(self, midi_data: Dict) -> Optional[np.ndarray]:
        """Convert MIDI data to audio."""
        try:
            if self.synthesizer:
                # Use FluidSynth for high-quality synthesis
                return self._fluidsynth_midi_to_audio(midi_data)
            else:
                # Use simple sine wave synthesis
                return self._simple_midi_to_audio(midi_data)
        
        except Exception as e:
            print(f"MIDI to audio conversion error: {e}")
            return None
    
    def _fluidsynth_midi_to_audio(self, midi_data: Dict) -> np.ndarray:
        """Convert MIDI to audio using FluidSynth."""
        # This is a simplified implementation
        # In practice, you'd need to properly handle MIDI events
        duration = 5.0  # Default duration
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Generate a simple melody
        frequencies = [440, 494, 523, 587, 659, 698, 784, 880]  # A4 to A5
        audio = np.zeros_like(t)
        
        for i, freq in enumerate(frequencies):
            start_time = i * duration / len(frequencies)
            end_time = (i + 1) * duration / len(frequencies)
            mask = (t >= start_time) & (t < end_time)
            audio[mask] += 0.1 * np.sin(2 * np.pi * freq * t[mask])
        
        return audio
    
    def _simple_midi_to_audio(self, midi_data: Dict) -> np.ndarray:
        """Convert MIDI to audio using simple synthesis."""
        # Extract notes from MIDI data
        events = midi_data.get('events', [])
        
        if not events:
            return np.zeros(int(self.sample_rate * 2))  # 2 seconds of silence
        
        # Calculate total duration
        max_time = max(event.get('time', 0) for event in events) + 1.0
        duration = min(max_time, 10.0)  # Cap at 10 seconds
        
        # Generate audio
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t)
        
        for event in events:
            if event.get('type') == 'note_on':
                pitch = event.get('pitch', 60)
                velocity = event.get('velocity', 64)
                start_time = event.get('time', 0)
                
                # Convert MIDI pitch to frequency
                frequency = 440 * (2 ** ((pitch - 69) / 12))
                
                # Generate note
                note_duration = 0.5  # Default note duration
                note_start = int(start_time * self.sample_rate)
                note_end = min(note_start + int(note_duration * self.sample_rate), len(t))
                
                if note_start < len(t) and note_end > note_start:
                    note_t = t[note_start:note_end] - start_time
                    note_audio = (velocity / 127.0) * 0.1 * np.sin(2 * np.pi * frequency * note_t)
                    
                    # Apply envelope
                    envelope = np.exp(-note_t * 3)  # Exponential decay
                    note_audio *= envelope
                    
                    audio[note_start:note_end] += note_audio
        
        return audio
    
    def _play_audio(self, audio_data: np.ndarray):
        """Play audio data."""
        try:
            sd.play(audio_data, samplerate=self.sample_rate, device=self.output_device)
        except Exception as e:
            print(f"Audio playback error: {e}")
    
    def start_playback(self):
        """Start audio playback."""
        self.is_playing = True
    
    def stop_playback(self):
        """Stop audio playback."""
        self.is_playing = False
        sd.stop()
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_playback()
        if self.synthesizer:
            self.synthesizer.delete()


def list_audio_devices():
    """List available audio devices."""
    try:
        devices = sd.query_devices()
        print("Available audio devices:")
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']} ({device['max_input_channels']} in, {device['max_output_channels']} out)")
    except Exception as e:
        print(f"Error listing audio devices: {e}")


def create_real_time_processor(model_path: str, **kwargs) -> RealTimeProcessor:
    """Create a real-time processor with default settings."""
    return RealTimeProcessor(model_path=model_path, **kwargs)
