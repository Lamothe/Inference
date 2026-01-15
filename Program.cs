using Llama.Backends;
using Llama.Backends.Cpu;
using Llama.Backends.Rocm;
using Llama.Models;
using Llama.Models.Llama2;
using Llama.Models.Llama3;

string checkpointPath = "stories110M.bin";
string? tokeniserPath = null;
float temperature = 1.0f;
float topp = 0.9f;
int steps = 256;
string? prompt = null;
ulong seed = 0;
string mode = "generate";
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
        case "-s": seed = ulong.Parse(args[i + 1]); break;
        case "-n": steps = int.Parse(args[i + 1]); break;
        case "-i": prompt = args[i + 1]; break;
        case "-k": tokeniserPath = args[i + 1]; break;
        case "-m": mode = args[i + 1]; break;
        case "-b": requestedBackend = args[i + 1]; break;
        default: throw new Exception($"Unknown parameter argument {args[i]}");
    }
}

if (seed <= 0)
{
    seed = (ulong)DateTime.Now.Ticks;
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

if (steps == 0 || steps > transformer.Config.seq_len)
{
    steps = transformer.Config.seq_len;
}

if (transformer.ModelVersion == ModelVersion.Llama2)
{
    var tokeniser = new LLama2Tokeniser(tokeniserPath ?? "Models/Llama2/tokeniser.bin", transformer.Config.vocab_size);
    var sampler = new Llama2Sampler(transformer.Config.vocab_size, temperature, topp, seed);
    var model = new Llama2Model(backend);

    if (mode == "generate")
    {
        BenchmarkResults(() => model.Generate(transformer, tokeniser, sampler, prompt ?? string.Empty, steps));
    }
    else if (mode == "chat")
    {
        model.Chat(transformer, tokeniser, sampler, steps);
    }
    else
    {
        Console.Error.WriteLine("Unknown mode: " + mode);
    }
}
else if (transformer.ModelVersion == ModelVersion.Llama3)
{
    var tokeniser = new Llama3Tokeniser(tokeniserPath ?? "Models/Llama3/tokeniser.bin", transformer.Config.vocab_size);
    var sampler = new Llama3Sampler(transformer.Config.vocab_size, temperature, topp, seed);
    var model = new Llama3Model(backend);

    if (mode == "generate")
    {
        BenchmarkResults(() => model.Generate(transformer, tokeniser, sampler, prompt ?? string.Empty, steps));
    }
    else if (mode == "chat")
    {
        model.Chat(transformer, tokeniser, sampler, steps);
    }
    else
    {
        Console.Error.WriteLine("Unknown mode: " + mode);
    }
}
else
{
    Console.Error.WriteLine("Unknown model version");
}

static long TimeInMs() => DateTimeOffset.Now.ToUnixTimeMilliseconds();

static void BenchmarkResults(Func<int> func)
{
    var start = TimeInMs();
    var tokens = func();
    long end = TimeInMs();
    Console.Error.WriteLine($"Token/s: {(tokens - 1) / (double)(end - start) * 1000}");
}
