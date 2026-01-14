using Llama.Backends;

namespace Llama.Models.Llama3;

public unsafe class Llama3Transformer : Transformer
{
    private readonly IBackend Backend;

    public Llama3Transformer(string checkpointPath, IBackend backend)
    {
        Backend = backend;

        Weights = new TransformerWeights();
        using var fs = new FileStream(checkpointPath, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);

        // Read Config
        Config = new ModelConfig
        {
            dim = br.ReadInt32(),
            hidden_dim = br.ReadInt32(),
            n_layers = br.ReadInt32(),
            n_heads = br.ReadInt32(),
            n_kv_heads = br.ReadInt32(),
            vocab_size = br.ReadInt32(),
            seq_len = br.ReadInt32()
        };

        var sharedWeights = Config.vocab_size > 0;
        Config.vocab_size = Math.Abs(Config.vocab_size);

        State = new RunState(Config, backend);

        long headerSize = br.BaseStream.Position;
        long totalFileSize = br.BaseStream.Length;
        long weightsSizeInBytes = totalFileSize - headerSize;

        // Allocate ONE giant lump of unmanaged memory
        // Store this pointer in your class so you can Free() it later!
        void* masterMemoryBlock = System.Runtime.InteropServices.NativeMemory.Alloc((nuint)weightsSizeInBytes);

        // Read the file content directly into this memory
        // We have to loop because Stream.Read only accepts 'int' (max 2GB per read)
        // but models can be much larger.
        {
            byte* writePtr = (byte*)masterMemoryBlock;
            long bytesRemaining = weightsSizeInBytes;
            Stream stream = br.BaseStream;

            // Use a Span to write directly to unmanaged memory
            while (bytesRemaining > 0)
            {
                // Read up to 1GB at a time (safe margin below int.MaxValue)
                int chunkSize = (int)Math.Min(bytesRemaining, 1024 * 1024 * 1024);
                var destSpan = new Span<byte>(writePtr, chunkSize);

                int read = stream.Read(destSpan);
                if (read == 0) throw new EndOfStreamException("Unexpected end of file while reading weights.");

                bytesRemaining -= read;
                writePtr += read;
            }
        }

        // Set the pointers (Pointer Arithmetic)
        // We now walk through the memory we just loaded.
        float* ptr = (float*)masterMemoryBlock;

        // Helpers for calculating offsets
        long n_layers = Config.n_layers;
        long dim = Config.dim;
        long head_size = dim / Config.n_heads;
        long kv_heads = Config.n_kv_heads;
        long hidden_dim = Config.hidden_dim;

        Weights.token_embedding_table = ptr;
        ptr += (long)Config.vocab_size * dim;

        Weights.rms_att_weight = ptr;
        ptr += n_layers * dim;

        Weights.wq = ptr;
        ptr += n_layers * dim * (Config.n_heads * head_size);

        Weights.wk = ptr;
        ptr += n_layers * dim * (kv_heads * head_size);

        Weights.wv = ptr;
        ptr += n_layers * dim * (kv_heads * head_size);

        Weights.wo = ptr;
        ptr += n_layers * (Config.n_heads * head_size) * dim;

        Weights.rms_ffn_weight = ptr;
        ptr += n_layers * dim;

        Weights.w1 = ptr;
        ptr += n_layers * dim * hidden_dim;

        Weights.w2 = ptr;
        ptr += n_layers * hidden_dim * dim;

        Weights.w3 = ptr;
        ptr += n_layers * dim * hidden_dim;

        Weights.rms_final_weight = ptr;
        ptr += dim;

        // Skip RoPE frequencies (legacy artifact in .bin files)
        // C code skips: seq_len * head_size / 2 (real) + seq_len * head_size / 2 (imag)
        ptr += (long)Config.seq_len * head_size;

        // Handle Classifier Weights
        if (sharedWeights)
        {
            Weights.wcls = Weights.token_embedding_table;
            // Note: We don't increment ptr here because we are reusing the earlier address
        }
        else
        {
            Weights.wcls = ptr;
            ptr += (long)Config.vocab_size * dim;
        }
    }

    public float* Forward(int token, int position)
    {
        int dim = Config.dim;
        int kv_dim = Config.dim * Config.n_kv_heads / Config.n_heads;
        int kv_mul = Config.n_heads / Config.n_kv_heads;
        int hidden_dim = Config.hidden_dim;
        int head_size = dim / Config.n_heads;

        // Copy token embedding into x
        System.Buffer.MemoryCopy(Weights.token_embedding_table + token * dim, State.x, dim * sizeof(float), dim * sizeof(float));

        for (int l = 0; l < Config.n_layers; l++)
        {
            // Attention RMSNorm
            Backend.RmsNorm(State.xb, State.x, Weights.rms_att_weight + l * dim, dim);

            // Key/Value cache offsets
            long loff = (long)l * Config.seq_len * kv_dim;

            // QKV Matmuls
            // s.q = wq @ xb
            Backend.MatMul(State.q, State.xb, Weights.wq + (long)l * dim * dim, dim, dim);
            // k_cache[pos] = wk @ xb (write directly to cache)
            State.k = State.keyCache + loff + position * kv_dim;
            Backend.MatMul(State.k, State.xb, Weights.wk + (long)l * dim * kv_dim, dim, kv_dim);
            // v_cache[pos] = wv @ xb
            State.v = State.valueCache + loff + position * kv_dim;
            Backend.MatMul(State.v, State.xb, Weights.wv + (long)l * dim * kv_dim, dim, kv_dim);

            // RoPE
            for (int i = 0; i < Config.n_heads; i++)
            {
                for (int j = 0; j < head_size; j += 2)
                {
                    float freq = 1.0f / MathF.Pow(500000.0f, (float)j / head_size);
                    float val = position * freq;
                    float fcr = MathF.Cos(val);
                    float fci = MathF.Sin(val);

                    // Rotate Q
                    int q_idx = i * head_size + j;
                    float q0 = State.q[q_idx];
                    float q1 = State.q[q_idx + 1];
                    State.q[q_idx] = q0 * fcr - q1 * fci;
                    State.q[q_idx + 1] = q0 * fci + q1 * fcr;

                    // Rotate K
                    if (i < Config.n_kv_heads)
                    {
                        int k_idx = i * head_size + j; // indexing into the current k_span (size kv_dim)
                        float k0 = State.k[k_idx];
                        float k1 = State.k[k_idx + 1];
                        State.k[k_idx] = k0 * fcr - k1 * fci;
                        State.k[k_idx + 1] = k0 * fci + k1 * fcr;
                    }
                }
            }

            // Multihead Attention
            Parallel.For(0, Config.n_heads, h =>
            {
                // Q vector for this head
                int q_offset = h * head_size;
                // Attention scores for this head
                int att_offset = h * Config.seq_len;

                // Iterate over all timesteps
                for (int t = 0; t <= position; t++)
                {
                    // Get K vector for this head at timestep t
                    // Offset: layer_offset + time_offset + head_offset
                    int k_head_offset = (h / kv_mul) * head_size;
                    long k_ptr_offset = loff + t * kv_dim + k_head_offset;

                    float score = 0.0f;
                    float* q_ptr = State.q + q_offset;
                    float* k_ptr = State.keyCache + k_ptr_offset;
                    for (int i = 0; i < head_size; i++)
                    {
                        score += q_ptr[i] * k_ptr[i];
                    }
                    score /= MathF.Sqrt(head_size);
                    State.att[att_offset + t] = score;
                }

                // Softmax
                Backend.Softmax(State.att + att_offset, position + 1);

                // Weighted sum of values -> xb
                int xb_offset = h * head_size;
                // Clear xb slice
                float* xb_ptr = State.xb + xb_offset;
                for (int i = 0; i < head_size; i++) xb_ptr[i] = 0f;

                for (int t = 0; t <= position; t++)
                {
                    int v_head_offset = (h / kv_mul) * head_size;
                    long v_ptr_offset = loff + t * kv_dim + v_head_offset;
                    float a = State.att[att_offset + t];
                    float* v_ptr = State.valueCache + v_ptr_offset;

                    for (int i = 0; i < head_size; i++)
                    {
                        xb_ptr[i] += a * v_ptr[i];
                    }
                }
            });

            // Final attention matmul
            Backend.MatMul(State.xb2, State.xb, Weights.wo + (long)l * dim * dim, dim, dim);

            // Residual connection
            for (int i = 0; i < dim; i++)
            {
                State.x[i] += State.xb2[i];
            }

            // FFN RMSNorm
            Backend.RmsNorm(State.xb, State.x, Weights.rms_ffn_weight + l * dim, dim);

            // FFN Matmuls
            Backend.MatMul(State.hb, State.xb, Weights.w1 + (long)l * dim * hidden_dim, hidden_dim, dim);
            Backend.MatMul(State.hb2, State.xb, Weights.w3 + (long)l * dim * hidden_dim, hidden_dim, dim);

            // SwiGLU
            for (int i = 0; i < hidden_dim; i++)
            {
                float val = State.hb[i];
                val *= (1.0f / (1.0f + MathF.Exp(-val))); // silu
                val *= State.hb2[i];
                State.hb[i] = val;
            }

            // Final FFN Matmul
            Backend.MatMul(State.xb, State.hb, Weights.w2 + (long)l * dim * hidden_dim, dim, hidden_dim);

            // Residual
            for (int i = 0; i < dim; i++) State.x[i] += State.xb[i];
        }

        // Final RMSNorm
        Backend.RmsNorm(State.x, State.x, Weights.rms_final_weight, dim);

        // Classifier
        Backend.MatMul(State.logits, State.x, Weights.wcls, Config.vocab_size, dim);

        return State.logits;
    }
}
