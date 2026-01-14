using Llama.Backends;

namespace Llama.Models.Llama3;

public unsafe class Llama3Transformer : Transformer
{
    public Llama3Transformer(string checkpointPath, IBackend backend)
    {
        this.Backend = backend;

        Weights = new TransformerWeights();
        State = new RunState();

        using var fs = new FileStream(checkpointPath, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);

        // Read Config
        Config = new Config
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

        // Read Weights
        int dim = Config.dim;
        int hidden_dim = Config.hidden_dim;
        int n_layers = Config.n_layers;
        int head_size = dim / Config.n_heads;
        int kv_heads = Config.n_kv_heads;

        static float[] ReadFloats(long count)
        {
            var bytes = br.ReadBytes((int)(count * sizeof(float)));
            var floats = new float[count];
            Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
            return floats;
        }

        Weights.token_embedding_table = ReadFloats(Config.vocab_size * dim);
        Weights.rms_att_weight = ReadFloats(n_layers * dim);
        Weights.wq = ReadFloats(n_layers * dim * (Config.n_heads * head_size));
        Weights.wk = ReadFloats(n_layers * dim * (kv_heads * head_size));
        Weights.wv = ReadFloats(n_layers * dim * (kv_heads * head_size));
        Weights.wo = ReadFloats(n_layers * (Config.n_heads * head_size) * dim);
        Weights.rms_ffn_weight = ReadFloats(n_layers * dim);
        Weights.w1 = ReadFloats(n_layers * dim * hidden_dim);
        Weights.w2 = ReadFloats(n_layers * hidden_dim * dim);
        Weights.w3 = ReadFloats(n_layers * dim * hidden_dim);
        Weights.rms_final_weight = ReadFloats(dim);

        // Skip RoPE freq_cis_real and freq_cis_imag (precomputed in C, but usually skipped in inference if computed on fly)
        // The C code skips: p->seq_len * head_size / 2; (twice)
        var ropeChunk = Config.seq_len * head_size / 2;
        fs.Seek(ropeChunk * sizeof(float) * 2, SeekOrigin.Current);

        Weights.wcls = sharedWeights ? Weights.token_embedding_table : ReadFloats(Config.vocab_size * dim);

        State.Initialize(Config);
    }

    public float[] Forward(int token, int position)
    {
        int dim = Config.dim;
        int kv_dim = (Config.dim * Config.n_kv_heads) / Config.n_heads;
        int kv_mul = Config.n_heads / Config.n_kv_heads;
        int hidden_dim = Config.hidden_dim;
        int head_size = dim / Config.n_heads;

        // Copy token embedding into x
        new Span<float>(Weights.token_embedding_table, token * dim, dim).CopyTo(State.x);

        for (int l = 0; l < Config.n_layers; l++)
        {
            // Attention RMSNorm
            backend.RmsNorm(State.xb, State.x, new Span<float>(Weights.rms_att_weight, l * dim, dim));

            // Key/Value cache offsets
            int loff = l * Config.seq_len * kv_dim;

            // QKV Matmuls
            // s.q = wq @ xb
            backend.MatMul(State.q, State.xb, new Span<float>(Weights.wq, l * dim * dim, dim * dim), dim, dim);
            // k_cache[pos] = wk @ xb (write directly to cache)
            // Note: In C code s->k pointer is set to cache position. Here we map via Span.
            var k_span = new Span<float>(State.key_cache, loff + position * kv_dim, kv_dim);
            backend.MatMul(k_span, State.xb, new Span<float>(Weights.wk, l * dim * kv_dim, dim * kv_dim), dim, kv_dim);
            // v_cache[pos] = wv @ xb
            var v_span = new Span<float>(State.value_cache, loff + position * kv_dim, kv_dim);
            backend.MatMul(v_span, State.xb, new Span<float>(Weights.wv, l * dim * kv_dim, dim * kv_dim), dim, kv_dim);

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
                        float k0 = k_span[k_idx];
                        float k1 = k_span[k_idx + 1];
                        k_span[k_idx] = k0 * fcr - k1 * fci;
                        k_span[k_idx + 1] = k0 * fci + k1 * fcr;
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
                    int k_ptr_offset = loff + t * kv_dim + k_head_offset;

                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++)
                    {
                        score += State.q[q_offset + i] * State.key_cache[k_ptr_offset + i];
                    }
                    score /= MathF.Sqrt(head_size);
                    State.att[att_offset + t] = score;
                }

                // Softmax
                Softmax(new Span<float>(State.att, att_offset, position + 1), position + 1);

                // Weighted sum of values -> xb
                int xb_offset = h * head_size;
                // Clear xb slice
                Array.Clear(State.xb, xb_offset, head_size);

                for (int t = 0; t <= position; t++)
                {
                    int v_head_offset = (h / kv_mul) * head_size;
                    int v_ptr_offset = loff + t * kv_dim + v_head_offset;
                    float a = State.att[att_offset + t];

                    for (int i = 0; i < head_size; i++)
                    {
                        State.xb[xb_offset + i] += a * State.value_cache[v_ptr_offset + i];
                    }
                }
            });

            // Final attention matmul
            backend.MatMul(State.xb2, State.xb, new Span<float>(Weights.wo, l * dim * dim, dim * dim), dim, dim);

            // Residual connection
            for (int i = 0; i < dim; i++)
            {
                State.x[i] += State.xb2[i];
            }

            // FFN RMSNorm
            backend.RmsNorm(State.xb, State.x, new Span<float>(Weights.rms_ffn_weight, l * dim, dim));

            // FFN Matmuls
            backend.MatMul(State.hb, State.xb, new Span<float>(Weights.w1, l * dim * hidden_dim, dim * hidden_dim), hidden_dim, dim);
            backend.MatMul(State.hb2, State.xb, new Span<float>(Weights.w3, l * dim * hidden_dim, dim * hidden_dim), hidden_dim, dim);

            // SwiGLU
            for (int i = 0; i < hidden_dim; i++)
            {
                float val = State.hb[i];
                val *= (1.0f / (1.0f + MathF.Exp(-val))); // silu
                val *= State.hb2[i];
                State.hb[i] = val;
            }

            // Final FFN Matmul
            backend.MatMul(State.xb, State.hb, new Span<float>(Weights.w2, l * dim * hidden_dim, hidden_dim * dim), dim, hidden_dim);

            // Residual
            for (int i = 0; i < dim; i++) State.x[i] += State.xb[i];
        }

        // Final RMSNorm
        backend.RmsNorm(State.x, State.x, new Span<float>(Weights.rms_final_weight, 0, dim));

        // Classifier
        backend.MatMul(State.logits, State.x, new Span<float>(Weights.wcls, 0, Config.vocab_size * dim), Config.vocab_size, dim);

        return State.logits;
    }
}
