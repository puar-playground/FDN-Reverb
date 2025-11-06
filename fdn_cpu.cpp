#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>

// CPU implementation of FDN reverb processing
// Much faster than Python loop due to compiled C++ code
std::vector<torch::Tensor> process_fdn_cpu(
    torch::Tensor buffers,      // (C, N, max_delay)
    torch::Tensor write_idx,    // (C, N)
    torch::Tensor lp_state,     // (C, N)
    torch::Tensor A,            // (N, N) - feedback matrix
    torch::Tensor frac_delays_all, // (T, N) - fractional delays for all samples
    torch::Tensor x,            // (C, T) - input
    float damp_factor,
    float damp_factor2,
    float wet_factor,
    float wet_factor2,
    float input_div,
    int max_delay,
    int N
) {
    // Get dimensions
    int C = x.size(0);
    int T = x.size(1);
    
    // Output tensor
    auto y_out = torch::zeros_like(x);
    
    // Get accessors for efficient element access
    auto buffers_acc = buffers.accessor<float, 3>();
    auto write_idx_acc = write_idx.accessor<int64_t, 2>();
    auto lp_state_acc = lp_state.accessor<float, 2>();
    auto A_acc = A.accessor<float, 2>();
    auto x_acc = x.accessor<float, 2>();
    auto y_out_acc = y_out.accessor<float, 2>();
    
    // Temporary storage for delay line reads
    std::vector<float> y_vec(N);
    std::vector<float> fb_vec(N);
    
    // Process each sample sequentially (feedback dependency)
    for (int t = 0; t < T; t++) {
        // Get fractional delays for this sample
        auto frac_delays = frac_delays_all[t];
        auto frac_delays_acc = frac_delays.accessor<float, 1>();
        
        // Process each channel
        for (int c = 0; c < C; c++) {
            // Read from all delay lines for this channel
            for (int n = 0; n < N; n++) {
                int64_t w_idx = write_idx_acc[c][n];
                
                // Calculate read position
                float read_pos = fmodf((float)w_idx - frac_delays_acc[n] + max_delay, (float)max_delay);
                if (read_pos < 0.0f) read_pos += max_delay;
                
                // Interpolation indices
                int i0 = ((int)floorf(read_pos)) % max_delay;
                int i1 = (i0 + 1) % max_delay;
                float frac = read_pos - floorf(read_pos);
                
                // Linear interpolation
                float y0 = buffers_acc[c][n][i0];
                float y1 = buffers_acc[c][n][i1];
                y_vec[n] = y0 + frac * (y1 - y0);
            }
            
            // Feedback mixing: fb = A @ y_vec
            float wet_sum = 0.0f;
            for (int n = 0; n < N; n++) {
                float fb_val = 0.0f;
                for (int m = 0; m < N; m++) {
                    fb_val += A_acc[n][m] * y_vec[m];
                }
                
                // Damping (one-pole lowpass)
                lp_state_acc[c][n] = lp_state_acc[c][n] * damp_factor + fb_val * damp_factor2;
                float fb_damped = lp_state_acc[c][n];
                
                // Input distribution
                float in_val = x_acc[c][t] * input_div;
                
                // Write back to delay buffer
                int64_t w_idx = write_idx_acc[c][n];
                int write_pos = w_idx % max_delay;
                buffers_acc[c][n][write_pos] = fb_damped + in_val;
                write_idx_acc[c][n] = (w_idx + 1) % max_delay;
                
                // Accumulate for output (mean over delay lines)
                wet_sum += y_vec[n];
            }
            
            // Output: wet/dry mix
            float wet_sample = wet_sum / (float)N;
            y_out_acc[c][t] = x_acc[c][t] * wet_factor + wet_sample * wet_factor2;
        }
    }
    
    return {y_out, buffers, write_idx, lp_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("process_fdn_cpu", &process_fdn_cpu, "FDN reverb CPU processing (fast C++ implementation)");
}

