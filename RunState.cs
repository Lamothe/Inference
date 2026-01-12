using Llama.Backends;

namespace Llama;

public unsafe struct RunState : IDisposable
{
    // current wave of activations
    public float* x; // activation at current time stamp (dim,)
    public float* xb; // same, but inside a residual branch (dim,)
    public float* xb2; // an additional buffer just for convenience (dim,)
    public float* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    public float* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    public float* q; // query (dim,)
    public float* k; // Just a pointer (view), DO NOT ALLOCATE
    public float* v; // Just a pointer (view), DO NOT ALLOCATE
    public float* att; // buffer for scores/attention values (n_heads, seq_len)
    public float* logits; // output logits
                          // kv cache
    public float* keyCache;   // (layer, seq_len, dim)
    public float* valueCache; // (layer, seq_len, dim)

    public float* rope_freq;

    private readonly IBackend backend;

    public RunState(Config p, IBackend backend)
    {
        this.backend = backend;

        ulong dimBytes = (ulong)(p.dim * sizeof(float));
        ulong hiddenBytes = (ulong)(p.hidden_dim * sizeof(float));

        // Calculate correct KV cache size
        // Note: kv_dim might be smaller than dim if using Grouped Query Attention (GQA)
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        ulong kvBytes = (ulong)(p.n_layers * p.seq_len * kv_dim * sizeof(float));

        // Allocations
        x = backend.Allocate<float>(dimBytes);
        xb = backend.Allocate<float>(dimBytes);
        xb2 = backend.Allocate<float>(dimBytes);
        hb = backend.Allocate<float>(hiddenBytes);  // Note: used hiddenBytes
        hb2 = backend.Allocate<float>(hiddenBytes); // Note: used hiddenBytes
        q = backend.Allocate<float>(dimBytes);

        // k and v are skipped! They are just pointers into the cache.

        att = backend.Allocate<float>((ulong)(p.n_heads * p.seq_len * sizeof(float)));
        logits = backend.Allocate<float>((ulong)(p.vocab_size * sizeof(float)));

        // Huge buffers
        keyCache = backend.Allocate<float>(kvBytes);
        valueCache = backend.Allocate<float>(kvBytes);

        rope_freq = backend.Allocate<float>((ulong)(p.dim / 2 * sizeof(float)));
    }

    public void Dispose()
    {
        // Removed 'state.' prefix
        if (x != null) backend.Free(x);
        if (xb != null) backend.Free(xb);
        if (xb2 != null) backend.Free(xb2);
        if (hb != null) backend.Free(hb);
        if (hb2 != null) backend.Free(hb2);
        if (q != null) backend.Free(q);

        // DO NOT free k or v (they point into key_cache/value_cache)

        if (att != null) backend.Free(att);
        if (logits != null) backend.Free(logits);
        if (keyCache != null) backend.Free(keyCache);
        if (valueCache != null) backend.Free(valueCache);

        if (rope_freq != null) backend.Free(rope_freq);
    }
}
