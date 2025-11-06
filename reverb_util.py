import torch
import numpy as np
import os

# Try to load CPU C++ extension if available
_fdn_cpp_available = False
_fdn_cpp_module = None
try:
    from torch.utils.cpp_extension import load
    
    # Load CPU C++ extension (works on all platforms)
    try:
        _fdn_cpp_module = load(
            name="fdn_cpu",
            sources=["fdn_cpu.cpp"],
            extra_cflags=["-O3", "-march=native"],  # Optimize for current CPU
            verbose=False
        )
        _fdn_cpp_available = True
        print("[FDNReverb] CPU C++ extension loaded successfully - using fast C++ implementation")
    except Exception as e:
        print(f"[FDNReverb] CPU C++ extension failed: {e}, using Python fallback")
        _fdn_cpp_available = False
            
except Exception as e:
    print(f"[FDNReverb] C++ extension not available, using Python fallback: {e}")
    _fdn_cpp_available = False




def hadamard_orthogonal(n: int) -> torch.Tensor:
    """Generate an n x n Hadamard orthogonal matrix (n must be a power of 2)."""
    if n & (n - 1) != 0:
        raise ValueError("Hadamard size must be power of 2.")
    
    # Recursively construct Hadamard matrix
    H = torch.ones(1, 1, dtype=torch.float32)
    current_size = 1
    while current_size < n:
        # H = [[H, H], [H, -H]]
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
        current_size *= 2
    
    # Normalize
    return H[:n, :n] / torch.sqrt(torch.tensor(n, dtype=torch.float32))


def _process_fdn(
    buffers: torch.Tensor,  # (C, N, max_delay) for batch processing
    write_idx: torch.Tensor,  # (C, N)
    lp_state: torch.Tensor,  # (C, N)
    A: torch.Tensor,
    frac_delays_all: torch.Tensor,
    x: torch.Tensor,  # (C, T) for batch processing
    damp_factor: float,
    damp_factor2: float,
    wet_factor: float,
    wet_factor2: float,
    input_div: float,
    max_delay: int,
    N: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Processing - optimized loop with batch processing for multiple channels."""
    # Try C++ extension first if available
    if _fdn_cpp_available:
        try:
            # Call C++ function (all tensors are already on CPU)
            result = _fdn_cpp_module.process_fdn_cpu(
                buffers, write_idx, lp_state, A, frac_delays_all, x,
                damp_factor, damp_factor2, wet_factor, wet_factor2,
                input_div, max_delay, N
            )
            
            y_out = result[0]
            buffers = result[1]
            write_idx = result[2]
            lp_state = result[3]
            
            return y_out, lp_state, write_idx
        except Exception as e:
            print(f"[FDNReverb] C++ extension failed, falling back to Python: {e}")
    
    # Python fallback
    C, length = x.shape
    y_out = torch.zeros_like(x)  # (C, T)
    delay_indices = torch.arange(N, dtype=torch.long)
    ones_vec = torch.ones(N, dtype=torch.float32)
    
    # Optimized loop - pre-compute constants and reduce operations
    # Show progress periodically
    progress_interval = max(10000, length // 100)
    for n in range(length):
        if n % progress_interval == 0 and n > 0:
            print(f"Processing: {n}/{length} samples ({100*n/length:.1f}%)")
        # Get fractional delays for this sample: (N,)
        frac_delays = frac_delays_all[n]
        
        # Calculate read positions for all channels: (C, N)
        read_pos = (write_idx.float() - frac_delays.unsqueeze(0)) % max_delay
        read_pos = torch.where(read_pos < 0.0, read_pos + max_delay, read_pos)
        
        # Interpolation indices (optimized)
        read_pos_floor = read_pos.floor()
        i0 = read_pos_floor.long() % max_delay  # (C, N)
        i1 = (i0 + 1) % max_delay  # (C, N)
        frac = read_pos - read_pos_floor  # (C, N)
        
        # Read from all delay lines for all channels: (C, N)
        # buffers is (C, N, max_delay), i0 and i1 are (C, N)
        # Use gather for efficient batch indexing
        # For each (channel, delay_line) pair, gather from the delay buffer
        i0_64 = i0.to(torch.int64).unsqueeze(-1)  # (C, N, 1)
        i1_64 = i1.to(torch.int64).unsqueeze(-1)  # (C, N, 1)
        y0 = torch.gather(buffers, 2, i0_64).squeeze(-1)  # (C, N)
        y1 = torch.gather(buffers, 2, i1_64).squeeze(-1)  # (C, N)
        y_vec = torch.lerp(y0, y1, frac)  # (C, N)
        
        # Feedback mixing per channel: (C, N) -> (C, N)
        # A is (N, N), y_vec is (C, N), result is (C, N)
        fb = torch.einsum('ij,cj->ci', A, y_vec)  # More efficient than looping
        
        # Damping (in-place operations)
        lp_state = lp_state * damp_factor + fb * damp_factor2  # (C, N)
        fb_damped = lp_state
        
        # Input distribution (optimized): (C,) -> (C, N)
        in_vec = (x[:, n].unsqueeze(1) * input_div) * ones_vec.unsqueeze(0)  # (C, N)
        
        # Write back (optimized) - use scatter for efficient batch indexing
        write_pos = (write_idx % max_delay).to(torch.int64)  # (C, N)
        buffers.scatter_(2, write_pos.unsqueeze(-1), (fb_damped + in_vec).unsqueeze(-1))
        write_idx = (write_idx + 1) % max_delay
        
        # Output (optimized): mean over delay lines per channel
        wet_sample = y_vec.mean(dim=1)  # (C,)
        y_out[:, n] = x[:, n] * wet_factor + wet_sample * wet_factor2
    
    return y_out, lp_state, write_idx


class FDNReverb:
    """
    Simple FDN reverb algorithm (N delay lines + orthogonal feedback matrix).
    Supports: feedback, damping (lowpass for high-frequency decay), slight delay modulation (avoids metallic artifacts).
    PyTorch implementation with CUDA/CPU acceleration support.
    
    NOTE: FDN reverb is inherently sequential due to feedback loops - each sample depends on previous samples.
    This makes it slower than convolution-based reverb (which uses FFT and can be fully parallelized).
    For maximum speed, consider using convolution-based reverb instead, though it has different sonic characteristics.
    """
    def __init__(
        self,
        sr: int,
        delays_ms=(29, 37, 43, 53, 61, 71, 79, 89),  # Coprime/prime numbers provide better diffusion
        feedback_gain=0.78,           # 0~1, larger = longer tail; 1 ≈ infinite (use with caution)
        damp=0.3,                     # 0~1, feedback loop lowpass strength (faster high-frequency decay)
        wet=0.25,                     # 0~1, wet/dry ratio
        mod_depth_ms=1.2,             # Modulation depth for each delay line
        mod_rate_hz=0.2,               # Modulation rate
        output_gain=1.0              # Output gain scaling factor (1.0 = no change, >1.0 = louder)
    ):
        self.sr = sr
        self.N = len(delays_ms)
        self.device = torch.device("cpu")
        
        # Delay length (samples)
        self.delays = torch.tensor(
            [max(1, int(sr * d / 1000.0)) for d in delays_ms],
            dtype=torch.int32
        )
        self.max_delay = int(torch.max(self.delays).item() + sr * 0.01)  # Leave some room for modulation

        # Loop state: circular buffer for each delay line (using 3D tensor for batch processing)
        # Shape: (max_channels, N, max_delay) - we'll expand as needed
        self.max_channels = 2  # Support up to 2 channels (stereo) by default
        self.buffers = torch.zeros(self.max_channels, self.N, self.max_delay, dtype=torch.float32)
        self.write_idx = torch.zeros(self.max_channels, self.N, dtype=torch.int64)

        # Orthogonal feedback matrix (Hadamard), multiplied by feedback gain for stability
        H = hadamard_orthogonal(self.N)
        self.A = feedback_gain * H

        # Damping filter (first-order lowpass for each line)
        self.damp = float(damp)
        self.lp_state = torch.zeros(self.max_channels, self.N, dtype=torch.float32)

        # Wet/dry
        self.wet = float(wet)
        
        # Output gain (simple scaling factor)
        self.output_gain = float(output_gain)

        # Modulation (avoids metallic/comb artifacts)
        self.mod_depth = float(mod_depth_ms) * sr / 1000.0
        self.mod_rate = float(mod_rate_hz)
        self.mod_phase = 2 * np.pi * torch.rand(self.N, dtype=torch.float32)

    def process(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process audio signal x (torch.Tensor), returns wet/dry mixed signal.
        Supports mono (1D) and stereo/multi-channel (2D: C, T) input.
        """
        # Ensure input is a tensor
        if isinstance(x, torch.Tensor):
            x = x.cpu()
        else:
            x = torch.tensor(x, dtype=torch.float32)
        
        original_shape = x.shape
        x = x.float()
        
        # Handle dimensions
        if x.dim() == 1:
            # Mono input: (T,) -> process as single channel
            x = x.unsqueeze(0)  # (1, T)
            is_mono = True
        elif x.dim() == 2:
            # Stereo/multi-channel: (C, T)
            is_mono = False
        elif x.dim() == 3:
            # Batch format: (B, C, T) -> take first batch
            x = x[0]  # (C, T)
            is_mono = False
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        num_channels = x.shape[0]
        length = x.shape[1]
        
        # Time vector (for LFO) - same for all channels
        t = torch.arange(length, dtype=torch.float32) / self.sr
        
        # Pre-compute all LFO modulations: (length, N)
        phase_matrix = (
            2 * np.pi * self.mod_rate * t.unsqueeze(1) + 
            self.mod_phase.unsqueeze(0)
        )
        lfo_all = self.mod_depth * torch.sin(phase_matrix)
        
        # Pre-compute all fractional delays: (length, N)
        delays_float = self.delays.float().unsqueeze(0)
        frac_delays_all = delays_float + lfo_all
        frac_delays_all = torch.clamp(frac_delays_all, 1.0, float(self.max_delay - 2))
        
        # Expand buffers if needed for more channels
        if num_channels > self.buffers.shape[0]:
            # Expand buffers, write_idx, and lp_state to accommodate more channels
            old_channels = self.buffers.shape[0]
            self.buffers = torch.cat([
                self.buffers,
                torch.zeros(num_channels - old_channels, self.N, self.max_delay, 
                          dtype=torch.float32)
            ], dim=0)
            self.write_idx = torch.cat([
                self.write_idx,
                torch.zeros(num_channels - old_channels, self.N, dtype=torch.int64)
            ], dim=0)
            self.lp_state = torch.cat([
                self.lp_state,
                torch.zeros(num_channels - old_channels, self.N, dtype=torch.float32)
            ], dim=0)
            self.max_channels = num_channels
        
        # Process all channels in parallel as a batch
        buffers = self.buffers[:num_channels].clone()  # (C, N, max_delay)
        write_idx = self.write_idx[:num_channels].clone()  # (C, N)
        lp_state = self.lp_state[:num_channels].clone()  # (C, N)
        
        # Process all channels at once: x is (C, T)
        y_out, lp_state, write_idx = _process_fdn(
            buffers, write_idx, lp_state, self.A,
            frac_delays_all, x,  # Pass all channels at once
            float(1.0 - self.damp), float(self.damp),
            float(1.0 - self.wet), float(self.wet),
            float(1.0 / self.N),
            self.max_delay, self.N
        )
        
        # Update state
        self.buffers[:num_channels] = buffers
        self.lp_state[:num_channels] = lp_state
        self.write_idx[:num_channels] = write_idx
        
        # Apply output gain scaling (simple multiplication)
        if self.output_gain != 1.0:
            y_out = y_out * self.output_gain
        
        # Restore original shape
        if is_mono and len(original_shape) == 1:
            y_out = y_out.squeeze(0)  # (T,)
        elif len(original_shape) == 3:
            y_out = y_out.unsqueeze(0)  # (1, C, T)
        
        return y_out

