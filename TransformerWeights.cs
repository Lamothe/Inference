namespace Llama2;

public unsafe struct TransformerWeights
{
    public float* token_embedding_table;    // (vocab_size, dim)
                                            // weights for rmsnorms
    public float* rms_att_weight; // (layer, dim) rmsnorm weights
    public float* rms_ffn_weight; // (layer, dim)
                                  // weights for matmuls. note dim == n_heads * head_size
    public float* wq; // (layer, dim, n_heads * head_size)
    public float* wk; // (layer, dim, n_kv_heads * head_size)
    public float* wv; // (layer, dim, n_kv_heads * head_size)
    public float* wo; // (layer, n_heads * head_size, dim)
                      // weights for ffn
    public float* w1; // (layer, hidden_dim, dim)
    public float* w2; // (layer, dim, hidden_dim)
    public float* w3; // (layer, hidden_dim, dim)
                      // final rmsnorm
    public float* rms_final_weight; // (dim,)
                                    // (optional) classifier weights for the logits, on the last layer
    public float* wcls;
}