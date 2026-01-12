using Llama;
using Llama.Backends;
using Llama.Backends.Cpu;
using Llama.Backends.Rocm;

string checkpointPath = "stories110M.bin";
string tokeniserPath = "tokeniser.bin";
float temperature = 1.0f;
float topp = 0.9f;
int steps = 256;
string? prompt = null;
ulong rngSeed = 0;
string mode = "generate";
string? systemPrompt = null;
string requestedBackend = "CPU";

// Argument parsing
for (int i = 0; i < args.Length; i += 2)
{
    if (i + 1 >= args.Length)
    {
        // If there's only one extra arg assume that it's the model file.
        checkpointPath = args[i];

        break;
    }

    switch (args[i])
    {
        case "-t": temperature = float.Parse(args[i + 1]); break;
        case "-p": topp = float.Parse(args[i + 1]); break;
        case "-s": rngSeed = ulong.Parse(args[i + 1]); break;
        case "-n": steps = int.Parse(args[i + 1]); break;
        case "-i": prompt = args[i + 1]; break;
        case "-k": tokeniserPath = args[i + 1]; break;
        case "-m": mode = args[i + 1]; break;
        case "-y": systemPrompt = args[i + 1]; break;
        case "-b": requestedBackend = args[i + 1]; break;
        default: throw new Exception($"Unknown parameter argument {args[i]}");
    }
}

if (rngSeed <= 0)
{
    rngSeed = (ulong)DateTime.Now.Ticks;
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

Console.WriteLine($"Initialising {requestedBackend} backend");

using IBackend backend = requestedBackend switch
{
    "CPU" => new CpuBackend(),
    "ROCm" => new RocmBackend(),
    _ => throw new Exception($"Unknown backend {requestedBackend}")
};

Console.WriteLine(backend.GetDescription());

using var transformer = new Transformer(checkpointPath, backend);

if (steps == 0 || steps > transformer.config.seq_len)
{
    steps = transformer.config.seq_len;
}

var tokeniser = new Tokeniser(tokeniserPath, transformer.config.vocab_size);
var sampler = new Sampler(transformer.config.vocab_size, temperature, topp, rngSeed);
var engine = new Engine(backend);

if (mode == "generate")
{
    engine.Generate(transformer, tokeniser, sampler, prompt!, steps);
}
else if (mode == "chat")
{
    engine.Chat(transformer, tokeniser, sampler, prompt, systemPrompt, steps);
}
else
{
    Console.Error.WriteLine("Unknown mode: " + mode);
}