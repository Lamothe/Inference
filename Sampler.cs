namespace Llama2;

public class Sampler
{
    public int vocab_size;
    public ProbIndex[] probindex;
    public float temperature;
    public float topp;
    public ulong rng_state;

    public Sampler(int vocab_size, float temperature, float topp, ulong rng_seed)
    {
        this.vocab_size = vocab_size;
        this.temperature = temperature;
        this.topp = topp;
        this.rng_state = rng_seed;
        this.probindex = new ProbIndex[vocab_size];
    }
}
