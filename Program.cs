using Llama;
using Llama.Backends;
using Llama.Backends.Cpu;
using Llama.Backends.Rocm;
using Llama.Models.Llama2;

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

using var transformer = new Models.Llama2.Llama2Transformer(checkpointPath, backend);

if (steps == 0 || steps > transformer.Config.seq_len)
{
    steps = transformer.Config.seq_len;
}

var tokeniser = new Models.Llama2.Llama2Tokeniser(tokeniserPath, transformer.Config.vocab_size);
var sampler = new Models.Llama2.Llama2Sampler(transformer.Config.vocab_size, temperature, topp, rngSeed);
var model = new Models.Llama2.Llama2Model(backend);

if (mode == "generate")
{
    var start = TimeInMs();
    var tokens = model.Generate(transformer, tokeniser, sampler, prompt!, steps);
    if (tokens > 1)
    {
        long end = TimeInMs();
        Console.Error.WriteLine($"Token/s: {(tokens - 1) / (double)(end - start) * 1000}");
    }
}
else if (mode == "chat")
{
    model.Chat(transformer, tokeniser, sampler, prompt, systemPrompt, steps);
}
else
{
    Console.Error.WriteLine("Unknown mode: " + mode);
}

static long TimeInMs() => DateTimeOffset.Now.ToUnixTimeMilliseconds();
