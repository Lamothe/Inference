using Llama2;

string checkpoint_path = "stories15M.bin";
string tokenizer_path = "tokenizer.bin";
float temperature = 1.0f;
float topp = 0.9f;
int steps = 256;
string? prompt = null;
ulong rng_seed = 0;
string mode = "generate";
string? system_prompt = null;

// Argument parsing
for (int i = 0; i < args.Length; i += 2)
{
    if (i + 1 >= args.Length)
    {
        break;
    }

    switch (args[i])
    {
        case "-t": temperature = float.Parse(args[i + 1]); break;
        case "-p": topp = float.Parse(args[i + 1]); break;
        case "-s": rng_seed = ulong.Parse(args[i + 1]); break;
        case "-n": steps = int.Parse(args[i + 1]); break;
        case "-i": prompt = args[i + 1]; break;
        case "-z": tokenizer_path = args[i + 1]; break;
        case "-m": mode = args[i + 1]; break;
        case "-y": system_prompt = args[i + 1]; break;
    }
}

if (rng_seed <= 0)
{
    rng_seed = (ulong)DateTime.Now.Ticks;
}

if (temperature < 0.0f)
{
    temperature = 0.0f;
}

if (topp < 0.0f || 1.0f < topp)
{
    topp = 0.9f;
}

if (steps < 0)
{
    steps = 0;
}

using var transformer = new Transformer();
Engine.BuildTransformer(transformer, checkpoint_path);

if (steps == 0 || steps > transformer.config.seq_len)
{
    steps = transformer.config.seq_len;
}

var tokenizer = new Tokenizer();
Engine.BuildTokenizer(tokenizer, tokenizer_path, transformer.config.vocab_size);

var sampler = new Sampler(transformer.config.vocab_size, temperature, topp, rng_seed);

if (mode == "generate")
{
    Engine.Generate(transformer, tokenizer, sampler, prompt!, steps);
}
else if (mode == "chat")
{
    Engine.Chat(transformer, tokenizer, sampler, prompt, system_prompt, steps);
}
else
{
    Console.Error.WriteLine("Unknown mode: " + mode);
}