using Llama.Backends;

namespace Llama.Models;

public unsafe class Transformer
{
    public ModelConfig Config;
    public TransformerWeights Weights;
    public RunState State;

    public float* Data;
    public ulong TotalByteSize;

    public required IBackend Backend;
}