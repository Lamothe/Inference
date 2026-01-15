using System.Numerics.Tensors;
using Llama.Backends;

namespace Llama.Models;

public unsafe class Transformer : IDisposable
{
    private readonly IBackend Backend;

    public ModelConfig Config;
    public TransformerWeights Weights;
    public RunState State;

    public float* Data;
    public ulong TotalByteSize;

    public ModelVersion ModelVersion { get; private set; }

    public Transformer(string checkpointPath, IBackend backend)
    {
        Backend = backend;

        using var fs = new FileStream(checkpointPath, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);

        var configBytes = new byte[sizeof(ModelConfig)];
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

        // Inside LoadModel, after reading the Config:
        if (Config.vocab_size == 32000)
        {
            ModelVersion = ModelVersion.Llama2;
            Console.WriteLine("[INFO] Detected Llama 2 architecture (Vocab: 32k)");
        }
        else if (Config.vocab_size >= 128000)
        {
            ModelVersion = ModelVersion.Llama3;
            Console.WriteLine("[INFO] Detected Llama 3 architecture (Vocab: 128k+)");
        }
        else
        {
            ModelVersion = ModelVersion.Unknown;
            Console.WriteLine($"[WARN] Unknown architecture (Vocab: {Config.vocab_size})");
        }

        // Allocate Pinned Memory for the WHOLE file (Config + Weights)
        TotalByteSize = (ulong)fs.Length;
        Data = Backend.Allocate<float>(TotalByteSize);

        if (Data == null)
        {
            Console.Error.WriteLine("Failed to allocate pinned memory for model!");
            Environment.Exit(1);
        }

        // Read the file into the pinned memory
        Console.WriteLine($"Loading model ({TotalByteSize / 1024 / 1024} MB) into memory...");

        // Rewind to start to read everything including config (to keep offsets simple)
        fs.Position = 0;

        // Use a Span to read directly into the native memory
        var dataSpan = new Span<byte>(Data, (int)TotalByteSize);
        var bytesRead = fs.Read(dataSpan);

        if (bytesRead != (int)TotalByteSize)
        {
            Console.Error.WriteLine("Failed to read the complete model file!");
            Environment.Exit(1);
        }

        // Set up the weight pointers
        // Offset by config size (exactly like run.c)
        var weightsPointer = (float*)((byte*)Data + sizeof(ModelConfig));
        var headSize = Config.dim / Config.n_heads;
        var nLayers = Config.n_layers;

        Weights.token_embedding_table = weightsPointer;
        weightsPointer += Config.vocab_size * Config.dim;
        Weights.rms_att_weight = weightsPointer;
        weightsPointer += nLayers * Config.dim;
        Weights.wq = weightsPointer;
        weightsPointer += nLayers * Config.dim * Config.n_heads * headSize;
        Weights.wk = weightsPointer;
        weightsPointer += nLayers * Config.dim * Config.n_kv_heads * headSize;
        Weights.wv = weightsPointer;
        weightsPointer += nLayers * Config.dim * Config.n_kv_heads * headSize;
        Weights.wo = weightsPointer;
        weightsPointer += nLayers * Config.n_heads * headSize * Config.dim;
        Weights.rms_ffn_weight = weightsPointer;
        weightsPointer += nLayers * Config.dim;
        Weights.w1 = weightsPointer;
        weightsPointer += nLayers * Config.dim * Config.hidden_dim;
        Weights.w2 = weightsPointer;
        weightsPointer += nLayers * Config.hidden_dim * Config.dim;
        Weights.w3 = weightsPointer;
        weightsPointer += nLayers * Config.dim * Config.hidden_dim;
        Weights.rms_final_weight = weightsPointer;
        weightsPointer += Config.dim;
        weightsPointer += Config.seq_len * headSize / 2;
        weightsPointer += Config.seq_len * headSize / 2;
        Weights.wcls = sharedWeights ? Weights.token_embedding_table : weightsPointer;

        State = new RunState(Config, backend);

        var dim = Config.dim;

        for (var i = 0; i < Config.dim; i += 2)
        {
            var headDim = i % headSize;
            var freq = 1.0f / MathF.Pow(10000.0f, headDim / (float)headSize);
            State.rope_freq[i / 2] = freq;
        }
    }

    public void Dispose()
    {
        State.Dispose();
        Backend.Free(Data);
        Data = null;
    }

    public float* Forward(int token, int position)
    {
        ModelConfig p = Config;
        TransformerWeights w = Weights;
        RunState s = State;
        float* x = s.x;
        int dim = p.dim;
        int kv_dim = p.dim * p.n_kv_heads / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads;
        int hidden_dim = p.hidden_dim;
        int head_size = dim / p.n_heads;

        // copy embedding
        float* content_row = w.token_embedding_table + token * dim;
        Buffer.MemoryCopy(content_row, x, dim * sizeof(float), dim * sizeof(float));

        for (int l = 0; l < p.n_layers; l++)
        {
            // attention rmsnorm
            Backend.RmsNorm(s.xb, x, w.rms_att_weight + l * dim, dim);

            // Calculate offsets using the outer 'kv_dim'
            long loff = (long)l * p.seq_len * kv_dim;
            long currentOffset = loff + (long)position * kv_dim;

            // SAFETY CHECK (Calculate limit locally)
            //    Total floats = Layers * SeqLen * KVDim
            long totalCacheElements = (long)p.n_layers * p.seq_len * kv_dim;

            if (currentOffset + kv_dim > totalCacheElements)
            {
                Console.WriteLine($"CRITICAL: KV Cache Overflow at Layer {l}, Pos {position}!");
                return null; // Stop safely
            }

            // Assign the pointers
            s.k = s.keyCache + currentOffset;
            s.v = s.valueCache + currentOffset;

            // qkv matmuls
            Backend.MatMul(s.q, s.xb, w.wq + (long)l * dim * dim, dim, dim);
            Backend.MatMul(s.k, s.xb, w.wk + (long)l * dim * kv_dim, dim, kv_dim);
            Backend.MatMul(s.v, s.xb, w.wv + (long)l * dim * kv_dim, dim, kv_dim);

            // RoPE
            for (int i = 0; i < dim; i += 2)
            {
                // Fast lookup, no division or modulo!
                float fcr = MathF.Cos(position * s.rope_freq[i / 2]);
                float fci = MathF.Sin(position * s.rope_freq[i / 2]);

                int rotn = i < kv_dim ? 2 : 1; // Existing logic
                for (int v = 0; v < rotn; v++)
                {
                    float* vec = v == 0 ? s.q : s.k;
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // Multihead attention
            Parallel.For(0, p.n_heads, h =>
            {
                float* q = s.q + h * head_size;
                float* att = s.att + h * p.seq_len;
                for (int t_step = 0; t_step <= position; t_step++)
                {
                    float* k = s.keyCache + loff + t_step * kv_dim + (h / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++)
                    {
                        score += q[i] * k[i];
                    }
                    score /= MathF.Sqrt(head_size);
                    att[t_step] = score;
                }

                Backend.Softmax(att, position + 1);

                float* xb = s.xb + h * head_size;
                // memset 0
                new Span<float>(xb, head_size).Clear();

                for (int t_step = 0; t_step <= position; t_step++)
                {
                    float* v = s.valueCache + loff + t_step * kv_dim + (h / kv_mul) * head_size;
                    float a = att[t_step];
                    for (int i = 0; i < head_size; i++)
                    {
                        xb[i] += a * v[i];
                    }
                }
            });

            // final matmul
            Backend.MatMul(s.xb2, s.xb, w.wo + (long)l * dim * dim, dim, dim);

            // residual connection back into x
            var x_span = new Span<float>(x, dim);
            var xb2_span = new ReadOnlySpan<float>(s.xb2, dim);

            // In-place addition: x = x + xb2
            TensorPrimitives.Add(x_span, xb2_span, x_span);

            // ffn rmsnorm
            Backend.RmsNorm(s.xb, x, w.rms_ffn_weight + l * dim, dim);

            // ffn
            Backend.MatMul(s.hb, s.xb, w.w1 + (long)l * dim * hidden_dim, dim, hidden_dim);
            Backend.MatMul(s.hb2, s.xb, w.w3 + (long)l * dim * hidden_dim, dim, hidden_dim);

            // SwiGLU
            Parallel.For(0, hidden_dim, i =>
            {
                float val = s.hb[i];
                val *= 1.0f / (1.0f + MathF.Exp(-val));
                val *= s.hb2[i];
                s.hb[i] = val;
            });

            Backend.MatMul(s.xb, s.hb, w.w2 + (long)l * dim * hidden_dim, hidden_dim, dim);

            // residual
            for (int i = 0; i < dim; i++) x[i] += s.xb[i];
        }

        Backend.RmsNorm(x, x, w.rms_final_weight, dim);
        Backend.MatMul(s.logits, x, w.wcls, p.dim, p.vocab_size);
        return s.logits;
    }
}