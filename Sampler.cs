namespace Llama;

public class Sampler(int vocab_size, float temperature, float topp, ulong rng_seed)
{
    public int vocab_size = vocab_size;
    public ProbIndex[] probindex = new ProbIndex[vocab_size];
    public float temperature = temperature;
    public float topp = topp;
    public ulong rng_state = rng_seed;
}
