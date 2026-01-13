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

    public float[] Forward(int token, int pos)
    {
        var p = Config;
        var w = Weights;
        var s = State;
        int dim = p.dim;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads;
        int hidden_dim = p.hidden_dim;
        int head_size = dim / p.n_heads;

        // Copy token embedding into x
        new Span<float>(w.token_embedding_table, token * dim, dim).CopyTo(s.x);

        for (int l = 0; l < p.n_layers; l++)
        {
            // Attention RMSNorm
            backend.RmsNorm(s.xb, s.x, new Span<float>(w.rms_att_weight, l * dim, dim));

            // Key/Value cache offsets
            int loff = l * p.seq_len * kv_dim;

            // QKV Matmuls
            // s.q = wq @ xb
            backend.MatMul(s.q, s.xb, new Span<float>(w.wq, l * dim * dim, dim * dim), dim, dim);
            // k_cache[pos] = wk @ xb (write directly to cache)
            // Note: In C code s->k pointer is set to cache position. Here we map via Span.
            var k_span = new Span<float>(s.key_cache, loff + pos * kv_dim, kv_dim);
            backend.MatMul(k_span, s.xb, new Span<float>(w.wk, l * dim * kv_dim, dim * kv_dim), dim, kv_dim);
            // v_cache[pos] = wv @ xb
            var v_span = new Span<float>(s.value_cache, loff + pos * kv_dim, kv_dim);
            backend.MatMul(v_span, s.xb, new Span<float>(w.wv, l * dim * kv_dim, dim * kv_dim), dim, kv_dim);

            // RoPE
            for (int i = 0; i < p.n_heads; i++)
            {
                for (int j = 0; j < head_size; j += 2)
                {
                    float freq = 1.0f / MathF.Pow(500000.0f, (float)j / head_size);
                    float val = pos * freq;
                    float fcr = MathF.Cos(val);
                    float fci = MathF.Sin(val);

                    // Rotate Q
                    int q_idx = i * head_size + j;
                    float q0 = s.q[q_idx];
                    float q1 = s.q[q_idx + 1];
                    s.q[q_idx] = q0 * fcr - q1 * fci;
                    s.q[q_idx + 1] = q0 * fci + q1 * fcr;

                    // Rotate K
                    if (i < p.n_kv_heads)
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
            Parallel.For(0, p.n_heads, h =>
            {
                // Q vector for this head
                int q_offset = h * head_size;
                // Attention scores for this head
                int att_offset = h * p.seq_len;

                // Iterate over all timesteps
                for (int t = 0; t <= pos; t++)
                {
                    // Get K vector for this head at timestep t
                    // Offset: layer_offset + time_offset + head_offset
                    int k_head_offset = (h / kv_mul) * head_size;
                    int k_ptr_offset = loff + t * kv_dim + k_head_offset;

                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++)
                    {
                        score += s.q[q_offset + i] * s.key_cache[k_ptr_offset + i];
                    }
                    score /= MathF.Sqrt(head_size);
                    s.att[att_offset + t] = score;
                }

                // Softmax
                Softmax(new Span<float>(s.att, att_offset, pos + 1), pos + 1);

                // Weighted sum of values -> xb
                int xb_offset = h * head_size;
                // Clear xb slice
                Array.Clear(s.xb, xb_offset, head_size);

                for (int t = 0; t <= pos; t++)
                {
                    int v_head_offset = (h / kv_mul) * head_size;
                    int v_ptr_offset = loff + t * kv_dim + v_head_offset;
                    float a = s.att[att_offset + t];

                    for (int i = 0; i < head_size; i++)
                    {
                        s.xb[xb_offset + i] += a * s.value_cache[v_ptr_offset + i];
                    }
                }
            });

            // Final attention matmul
            backend.MatMul(s.xb2, s.xb, new Span<float>(w.wo, l * dim * dim, dim * dim), dim, dim);

            // Residual connection
            for (int i = 0; i < dim; i++)
            {
                s.x[i] += s.xb2[i];
            }

            // FFN RMSNorm
            backend.RmsNorm(s.xb, s.x, new Span<float>(w.rms_ffn_weight, l * dim, dim));

            // FFN Matmuls
            backend.MatMul(s.hb, s.xb, new Span<float>(w.w1, l * dim * hidden_dim, dim * hidden_dim), hidden_dim, dim);
            backend.MatMul(s.hb2, s.xb, new Span<float>(w.w3, l * dim * hidden_dim, dim * hidden_dim), hidden_dim, dim);

            // SwiGLU
            for (int i = 0; i < hidden_dim; i++)
            {
                float val = s.hb[i];
                val *= (1.0f / (1.0f + MathF.Exp(-val))); // silu
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            // Final FFN Matmul
            backend.MatMul(s.xb, s.hb, new Span<float>(w.w2, l * dim * hidden_dim, hidden_dim * dim), dim, hidden_dim);

            // Residual
            for (int i = 0; i < dim; i++) s.x[i] += s.xb[i];
        }

        // Final RMSNorm
        backend.RmsNorm(s.x, s.x, new Span<float>(w.rms_final_weight, 0, dim));

        // Classifier
        backend.MatMul(s.logits, s.x, new Span<float>(w.wcls, 0, p.vocab_size * dim), p.vocab_size, dim);

        return s.logits;
    }
}
