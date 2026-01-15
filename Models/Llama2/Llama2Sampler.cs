using Llama.Backends;

namespace Llama.Models.Llama2;

public unsafe class Llama2Sampler(int vocab_size, float temperature, float topp, ulong rng_seed)
{
    public int vocab_size = vocab_size;
    public ProbIndex[] probindex = new ProbIndex[vocab_size];
    public float temperature = temperature;
    public float topp = topp;

    public int Sample(float* logits, IBackend backend)
    {
        if (temperature == 0.0f)
        {
            return backend.SampleArgmax(logits, vocab_size);
        }

        for (int q = 0; q < vocab_size; q++)
        {
            logits[q] /= temperature;
        }

        backend.Softmax(logits, vocab_size);

        float coin = backend.RandomF32(ref rng_seed);

        if (topp <= 0 || topp >= 1)
        {
            return backend.SampleMult(logits, vocab_size, coin);
        }
        else
        {
            return backend.SampleTopp(logits, vocab_size, topp, probindex, coin);
        }
    }
}
