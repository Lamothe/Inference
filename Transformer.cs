using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;

namespace Llama2;

public unsafe class Transformer : IDisposable
{
    public Config config; // the hyperparameters of the architecture (the blueprint)
    public TransformerWeights weights; // the weights of the model
    public RunState state; // buffers for the "wave" of activations in the forward pass

    // Memory mapping resources
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    public byte* _data; // pointer to the start of memory mapped data

    public void Dispose()
    {
        FreeRunState();
        if (_accessor != null)
        {
            _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            _accessor.Dispose();
        }
        _mmf?.Dispose();
    }

    public void MallocRunState()
    {
        int kv_dim = config.dim * config.n_kv_heads / config.n_heads;

        // Using NativeMemory (requires .NET 6+) for alignment and raw pointers
        // Zeroed memory (AllocZeroed) to match calloc
        state.x = (float*)NativeMemory.AllocZeroed((nuint)config.dim, sizeof(float));
        state.xb = (float*)NativeMemory.AllocZeroed((nuint)config.dim, sizeof(float));
        state.xb2 = (float*)NativeMemory.AllocZeroed((nuint)config.dim, sizeof(float));
        state.hb = (float*)NativeMemory.AllocZeroed((nuint)config.hidden_dim, sizeof(float));
        state.hb2 = (float*)NativeMemory.AllocZeroed((nuint)config.hidden_dim, sizeof(float));
        state.q = (float*)NativeMemory.AllocZeroed((nuint)config.dim, sizeof(float));
        state.key_cache = (float*)NativeMemory.AllocZeroed((nuint)(config.n_layers * config.seq_len * kv_dim), sizeof(float));
        state.value_cache = (float*)NativeMemory.AllocZeroed((nuint)(config.n_layers * config.seq_len * kv_dim), sizeof(float));
        state.att = (float*)NativeMemory.AllocZeroed((nuint)(config.n_heads * config.seq_len), sizeof(float));
        state.logits = (float*)NativeMemory.AllocZeroed((nuint)config.vocab_size, sizeof(float));
    }

    public void FreeRunState()
    {
        if (state.x != null) NativeMemory.Free(state.x);
        if (state.xb != null) NativeMemory.Free(state.xb);
        if (state.xb2 != null) NativeMemory.Free(state.xb2);
        if (state.hb != null) NativeMemory.Free(state.hb);
        if (state.hb2 != null) NativeMemory.Free(state.hb2);
        if (state.q != null) NativeMemory.Free(state.q);
        if (state.att != null) NativeMemory.Free(state.att);
        if (state.logits != null) NativeMemory.Free(state.logits);
        if (state.key_cache != null) NativeMemory.Free(state.key_cache);
        if (state.value_cache != null) NativeMemory.Free(state.value_cache);
    }

    public void ReadCheckpoint(string checkpointPath)
    {
        if (!File.Exists(checkpointPath))
        {
            Console.Error.WriteLine($"Couldn't open file {checkpointPath}");
            Environment.Exit(1);
        }

        using var fs = new FileStream(checkpointPath, FileMode.Open, FileAccess.Read);

        // Read config
        byte[] configBytes = new byte[sizeof(Config)];
        fs.ReadExactly(configBytes, 0, sizeof(Config));
        fixed (byte* pConfig = configBytes)
        {
            config = *(Config*)pConfig;
        }

        int shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = Math.Abs(config.vocab_size);
        long file_size = fs.Length;
        fs.Close();

        // Memory map
        _mmf = MemoryMappedFile.CreateFromFile(checkpointPath, FileMode.Open);
        _accessor = _mmf.CreateViewAccessor(0, file_size);
        byte* ptr = null;
        _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
        _data = ptr;

        // Offset by config size
        float* weights_ptr = (float*)(_data + sizeof(Config));
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
        ptr += n_layers * config.dim * (config.n_heads * head_size);
        weights.wk = ptr;
        ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
        weights.wv = ptr;
        ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
        weights.wo = ptr;
        ptr += n_layers * (config.n_heads * head_size) * config.dim;
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
        ptr += config.seq_len * head_size / 2; // skip freq_cis_real
        ptr += config.seq_len * head_size / 2; // skip freq_cis_imag
        weights.wcls = shared_weights != 0 ? weights.token_embedding_table : ptr;
    }
}
