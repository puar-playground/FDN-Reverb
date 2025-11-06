import torch
from reverb_util import FDNReverb
from io_utils import save_audio, load_audio
import os
import time
import argparse
os.system('clear')


def parse_args():
    parser = argparse.ArgumentParser(description='FDN Reverb Audio Processor')
    
    # Input/Output
    parser.add_argument('--input', '-i', type=str, default='dry.mp3',
                        help='Input audio file path')
    parser.add_argument('--output', '-o', type=str, default='wet_fdn.mp3',
                        help='Output audio file path')
    parser.add_argument('--mono', action='store_true',
                        help='Convert to mono (default: stereo)')
    parser.add_argument('--target_sample_rate', type=int, default=None,
                        help='Target sample rate (default: preserve original)')
    
    # FDN Reverb Parameters
    parser.add_argument('--delays_ms', type=float, nargs='+', 
                        default=[29, 37, 43, 53, 61, 71, 79, 89],
                        help='Delay times in milliseconds (default: 29 37 43 53 61 71 79 89)')
    parser.add_argument('--feedback_gain', type=float, default=0.9,
                        help='Feedback gain 0~1, larger = longer tail, must be <1 (default: 0.9)')
    parser.add_argument('--damp', type=float, default=0.25,
                        help='Damping factor 0~1, larger = faster high-frequency decay (default: 0.25)')
    parser.add_argument('--wet', type=float, default=1.0,
                        help='Wet/dry mix ratio 0~1, 0=dry, 1=fully wet (default: 1.0)')
    parser.add_argument('--mod_depth_ms', type=float, default=0.8,
                        help='Modulation depth in milliseconds (default: 0.8)')
    parser.add_argument('--mod_rate_hz', type=float, default=0.2,
                        help='Modulation rate in Hz (default: 0.2)')
    parser.add_argument('--output_gain', type=float, default=1.0,
                        help='Output gain scaling factor (default: 1.0)')
    
    # Volume matching
    parser.add_argument('--no_volume_match', action='store_true',
                        help='Disable automatic volume matching')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Read audio
    print(f"Loading audio from: {args.input}")
    audio, sr = load_audio(args.input, mono=args.mono, target_sample_rate=args.target_sample_rate)
    print(f"Audio shape: {audio.shape}, Sample rate: {sr} Hz")
    
    # Convert delays_ms to tuple
    delays_ms = tuple(args.delays_ms)
    
    # Create FDN reverb
    print(f"\nCreating FDN Reverb with parameters:")
    print(f"  Delays: {delays_ms} ms")
    print(f"  Feedback gain: {args.feedback_gain}")
    print(f"  Damping: {args.damp}")
    print(f"  Wet: {args.wet}")
    print(f"  Modulation: {args.mod_depth_ms} ms @ {args.mod_rate_hz} Hz")
    
    reverb = FDNReverb(
        sr=sr,
        delays_ms=delays_ms,
        feedback_gain=args.feedback_gain,
        damp=args.damp,
        wet=args.wet,
        mod_depth_ms=args.mod_depth_ms,
        mod_rate_hz=args.mod_rate_hz,
        output_gain=args.output_gain
    )
    
    # Process audio
    print(f"\nProcessing audio...")
    start_time = time.time()
    wet = reverb.process(audio)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Input shape: {audio.shape}, Output shape: {wet.shape}")
    
    # Volume matching
    if not args.no_volume_match:
        print(f"\nMatching volume levels...")
        input_peak = torch.mean(torch.abs(audio))
        output_peak = torch.mean(torch.abs(wet))
        
        if output_peak > 1e-6:
            volume_scale = input_peak / output_peak
            print(f"Input peak: {input_peak:.6f}, Output peak: {output_peak:.6f}")
            print(f"Volume scale: {volume_scale:.3f}")
            wet = wet * volume_scale
        else:
            print("Warning: Output peak too low, skipping volume scaling")
    
    # Save output
    print(f"\nSaving output to: {args.output}")
    save_audio(wet, sample_rate=sr, file_path=args.output)
    print("Done!")


if __name__ == "__main__":
    main()