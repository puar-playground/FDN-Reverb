from pathlib import Path
import shutil
from pydub import AudioSegment
import json
import torchaudio
import os
import torch
import torchaudio.transforms as T
import torchaudio.functional as AF
import re
import numpy as np
import subprocess
import tempfile

def resample(audio_wav, sr_in, sr_out):

    if sr_in != sr_out:
        audio_wav = AF.resample(waveform, orig_freq=sr_in, new_freq=sr_out)

    return audio_wav, sr_out

def convert_to_16k(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    if audio.frame_rate == 16000:
        # Already 16kHz — just copy the original
        shutil.copy2(input_path, output_path)
        # print(f"[✓] Copied (already 16kHz): {input_path}")
    else:
        # Resample and save as WAV
        audio = audio.set_frame_rate(16000)
        audio.export(output_path, format="wav")
        # print(f"[→] Converted to 16kHz: {input_path} → {output_path}")


def load_audio(file_path, target_sample_rate=None, min_duration=5, mono=False):
    waveform, sample_rate_origin = torchaudio.load(file_path)

    # Convert to mono if needed
    if mono:
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            # waveform = waveform[:1, :]

    # Resample if needed
    if target_sample_rate and sample_rate_origin != target_sample_rate:
        # waveform = T.Resample(orig_freq=sample_rate_origin, new_freq=target_sample_rate)(waveform)
        waveform = AF.resample(waveform, orig_freq=sample_rate_origin, new_freq=target_sample_rate)
        sample_rate = target_sample_rate
    else:
        sample_rate = sample_rate_origin
        
    # # Pad if too short
    # target_frames = int(min_duration * sample_rate)
    # if waveform.size(1) < target_frames:
    #     pad_amount = target_frames - waveform.size(1)
    #     waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    # Return (C, T) format directly - no batch dimension
    return waveform, sample_rate
    

def save_audio(waveform: torch.Tensor, sample_rate: int, file_path: str, target_sample_rate=None, bitrate=None):
    """
    Save a waveform tensor to a local audio file (e.g., .wav or .flac).

    Args:
        waveform (torch.Tensor): Shape (1, T) or (T,)
        sample_rate (int): Sampling rate in Hz
        file_path (str): Output file path (must end with .wav, .flac, etc.)
        target_sample_rate (int, optional): Resample to this sample rate before saving
        bitrate (str, optional): For MP3 files, specify bitrate (e.g., "128k", "192k", "256k", "320k")
    """
    # Remove batch dimension if present
    if waveform.dim() == 3 and waveform.size(0) == 1:
        waveform = waveform.squeeze(0)
    
    # Ensure shape is (channels, time)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach()

    # Resample if needed
    if target_sample_rate and sample_rate != target_sample_rate:
        # Ensure (C, T), float32, contiguous
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(torch.float32).contiguous()

        resampler = T.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
            resampling_method="kaiser_window",   # or "sinc_interpolation"
            lowpass_filter_width=64,             # increase to 128 if you still hear artifacts
            rolloff=0.9475937167,                # SOX default
            beta=14.7696564594                   # SOX default
        )
        waveform = resampler(waveform)

        # waveform = AF.resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)

        sample_rate = target_sample_rate
    
    # Handle MP3 with specific bitrate
    if file_path.split('/')[-1].lower().endswith('.mp3') and bitrate:
        print(f"[save_audio] Saving MP3 with bitrate: {bitrate}")
        
        # Parse numeric bitrate
        import re
        match = re.search(r"\d+(\.\d+)?", bitrate)
        parsed_bitrate = match.group() if match else "128"
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav") as f_temp:
            temp_wav_path = f_temp.name
            
            # Save as WAV first
            torchaudio.save(temp_wav_path, waveform, sample_rate)
            
            # Use FFmpeg to convert to MP3 with specific bitrate
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-i", temp_wav_path,
                "-ar", str(sample_rate),
                "-ac", str(waveform.size(0)),  # Number of channels
                "-b:a", f"{parsed_bitrate}k",
                "-c:a", "mp3",
                "-write_xing", "0",  # Disable VBR header
                "-id3v2_version", "0",  # Disable ID3 tags
                "-f", "mp3",
                file_path
            ]
            
            print(f"[save_audio] FFmpeg command: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                print(f"[save_audio] MP3 saved successfully: {file_path}")
                
                # Check file size
                import os
                file_size = os.path.getsize(file_path)
                print(f"[save_audio] File size: {file_size} bytes")
                
            except subprocess.CalledProcessError as e:
                print(f"[save_audio] FFmpeg error: {e}")
                print(f"[save_audio] FFmpeg stderr: {e.stderr}")
                raise RuntimeError(f"Failed to save MP3 with bitrate {bitrate}: {e}")
    
    else:
        torchaudio.save(file_path, waveform, sample_rate)



import os
from typing import Iterable, List, Tuple, Union

def find_files_by_extension(
    root_dir: str,
    extensions: Union[str, Iterable[str]]
) -> List[Tuple[str, str, str]]:
    """
    Search for all files whose extension is in `extensions` (case-insensitive)
    across all subfolders of `root_dir`.

    Args:
        root_dir: The root directory to search in.
        extensions: A string like ".mp3" or "mp3", or an iterable of such strings.

    Returns:
        A list of tuples (full_path, folder_path, file_name), sorted by full_path.
    """
    # Normalize to a tuple of lowercase extensions, each starting with "."
    if isinstance(extensions, str):
        extensions = [extensions]
    exts = tuple(
        (e if e.startswith(".") else f".{e}").lower()
        for e in extensions
    )

    result: List[Tuple[str, str, str]] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(exts):
                full_path = os.path.join(dirpath, file)
                result.append((full_path, dirpath, file))

    return sorted(result, key=lambda x: x[0])




def get_mp3(wav_tensor: torch.Tensor, sr: int, bitrate: str = "128k") -> torch.Tensor:
    """
    Convert a batch of audio tensors to MP3 format using FFmpeg,
    preserving stereo/multichannel structure and original length.

    Args:
        wav_tensor (torch.Tensor): Input audio tensor of shape (B, C, T).
        sr (int): Sample rate.
        bitrate (str): Bitrate for MP3 compression (e.g., '128k').

    Returns:
        torch.Tensor: MP3-compressed tensor with original shape (B, C, T).
    """
    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape
    flat_tensor = wav_tensor.contiguous().view(1, -1).cpu()

    # Parse numeric bitrate
    match = re.search(r"\d+(\.\d+)?", bitrate)
    parsed_bitrate = match.group() if match else "128"

    with tempfile.NamedTemporaryFile(suffix=".wav") as f_in, tempfile.NamedTemporaryFile(suffix=".mp3") as f_out:
        input_path, output_path = f_in.name, f_out.name

        # Save WAV
        torchaudio.save(input_path, flat_tensor, sr, backend="ffmpeg")

        # Build ffmpeg command for MP3 with explicit CBR settings
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(sr),  # Sample rate
            "-ac", str(channels),  # Number of channels
            "-b:a", f"{parsed_bitrate}k",  # Bitrate
            "-c:a", "mp3",  # MP3 codec
            "-write_xing", "0",  # Disable VBR header
            "-id3v2_version", "0",  # Disable ID3 tags
            "-f", "mp3",  # Force MP3 format
            output_path
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            mp3_tensor, loaded_sr = torchaudio.load(output_path, backend="ffmpeg")
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert using ffmpeg. Check ffmpeg installation and MP3 encoder.\n"
                f"Command: {' '.join(cmd)}"
            ) from e

    # Check if sample rate changed
    if loaded_sr != sr:
        print(f"Warning: Sample rate changed from {sr} to {loaded_sr}")
        mp3_tensor = AF.resample(mp3_tensor, orig_freq=loaded_sr, new_freq=sr)

    # Restore shape (B, C, T)
    mp3_tensor = mp3_tensor.to(device)
    compressed_length = mp3_tensor.size(-1)

    # Fix length mismatch
    target_length = batch_size * channels * original_length
    if compressed_length > target_length:
        mp3_tensor = mp3_tensor[:, :target_length]
    elif compressed_length < target_length:
        pad = torch.zeros(1, target_length - compressed_length, device=device)
        mp3_tensor = torch.cat([mp3_tensor, pad], dim=-1)

    output = mp3_tensor.view(batch_size, channels, -1)
    assert output.shape[-1] == original_length, (
        f"Shape mismatch after MP3 compression. Expected length {original_length}, "
        f"got {output.shape[-1]}. Make sure ffmpeg is installed and MP3 encoder is supported."
    )

    return output
