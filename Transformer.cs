using System.Runtime.InteropServices;
using Llama.Backends;

namespace Llama;

public unsafe class Transformer : IDisposable
{
    public Config config;
    public TransformerWeights weights;
    public RunState state;

    // Replaced MemoryMappedFile with a raw pinned pointer
    public float* _data;
    private ulong _totalByteSize;

    private IBackend backend;

    public Transformer(string checkpoint_path, IBackend backend)
    {
        this.backend = backend;

        ReadCheckpoint(checkpoint_path);
        state = new RunState(config, backend);
    }

    public void Dispose()
    {
        state.Dispose();

        if (_data != null)
        {
            backend.Free(_data);
            _data = null;
        }
    }

    public void ReadCheckpoint(string checkpointPath)
    {
        if (!File.Exists(checkpointPath))
        {
            Console.Error.WriteLine($"Couldn't open file {checkpointPath}");
            Environment.Exit(1);
        }

        using var fs = new FileStream(checkpointPath, FileMode.Open, FileAccess.Read);

        // 1. Read Config
        var configBytes = new byte[sizeof(Config)];
        fs.ReadExactly(configBytes, 0, sizeof(Config));
        fixed (byte* pConfig = configBytes)
        {
            config = *(Config*)pConfig;
        }

        var shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = Math.Abs(config.vocab_size);

        // Allocate Pinned Memory for the WHOLE file (Config + Weights)
        // This mirrors run.c logic exactly
        _totalByteSize = (ulong)fs.Length;
        _data = backend.Allocate<float>(_totalByteSize);

        if (_data == null)
        {
            Console.Error.WriteLine("Failed to allocate pinned memory for model!");
            Environment.Exit(1);
        }

        // 3. Read the file into the pinned memory
        Console.WriteLine($"Loading model ({_totalByteSize / 1024 / 1024} MB) into memory...");

        // Rewind to start to read everything including config (to keep offsets simple)
        fs.Position = 0;

        // Use a Span to read directly into the native memory
        var dataSpan = new Span<byte>(_data, (int)_totalByteSize);
        int bytesRead = fs.Read(dataSpan);

        if (bytesRead != (int)_totalByteSize)
        {
            Console.Error.WriteLine("Failed to read the complete model file!");
            Environment.Exit(1);
        }

        // 4. Set up the weight pointers
        // Offset by config size (exactly like run.c)
        var weights_ptr = (float*)((byte*)_data + sizeof(Config));
        MemoryMapWeights(weights_ptr, shared_weights);
    }

    private void MemoryMapWeights(float* ptr, int shared_weights)
    {
        int head_size = config.dim / config.n_heads;
        long n_layers = config.n_layers;

        weights.token_embedding_table = ptr;
        ptr += config.vocab_size * config.dim;
        weights.rms_att_weight = ptr;
        ptr += n_layers * config.dim;
        weights.wq = ptr;
        ptr += n_layers * config.dim * config.n_heads * head_size;
        weights.wk = ptr;
        ptr += n_layers * config.dim * config.n_kv_heads * head_size;
        weights.wv = ptr;
        ptr += n_layers * config.dim * config.n_kv_heads * head_size;
        weights.wo = ptr;
        ptr += n_layers * config.n_heads * head_size * config.dim;
        weights.rms_ffn_weight = ptr;
        ptr += n_layers * config.dim;
        weights.w1 = ptr;
        ptr += n_layers * config.dim * config.hidden_dim;
        weights.w2 = ptr;
        ptr += n_layers * config.hidden_dim * config.dim;
        weights.w3 = ptr;
        ptr += n_layers * config.dim * config.hidden_dim;
        weights.rms_final_weight = ptr;
        ptr += config.dim;
        ptr += config.seq_len * head_size / 2;
        ptr += config.seq_len * head_size / 2;
        weights.wcls = shared_weights != 0 ? weights.token_embedding_table : ptr;
    }
}