using Llama.Backends;

namespace Llama.Models.Llama3;

public unsafe class Llama3Model
{
    public void Generate(Llama3Transformer transformer, Llama3Tokeniser tokeniser, Llama3Sampler sampler, string prompt, int steps)
    {
        var tokens = tokeniser.Encode(prompt, true, false);
        int token = tokens.Length > 0 ? tokens[0] : 128000; // BOS if empty
        int pos = 0;
        int next;

        long start = 0;

        Console.Write(prompt);

        while (pos < steps)
        {
            float* logits = transformer.Forward(token, pos);

            if (pos < tokens.Length - 1)
            {
                next = tokens[pos + 1];
            }
            else
            {
                next = sampler.Sample(logits);
            }
            pos++;

            if (next == 128001 || next == 128009) break; // EOS

            string piece = tokeniser.Decode(next);
            Console.Write(piece);
            token = next;

            if (start == 0) start = DateTime.Now.Ticks;
        }
        Console.WriteLine();

        // Stats
        if (pos > 1)
        {
            long end = DateTime.Now.Ticks;
            double seconds = (end - start) / 10_000_000.0;
            Console.WriteLine($"Tokens per second: {(pos - 1) / seconds:F2}");
        }
    }

    public void Chat(Llama3Transformer transformer, Llama3Tokeniser tokeniser, Llama3Sampler s, int steps)
    {
        // Simple chat loop implementation mirroring the C code
        // Note: Managing chat history/KV-cache correctly requires state persistence
        // The C code resets or assumes continuous flow. This is a basic skeleton.
        Console.WriteLine("Chat mode (Type 'exit' to quit)");

        // For a true persistent chat, we'd need to manage the pos index and cache carefully
        // This example just runs one generation per prompt for simplicity, 
        // effectively resetting context.
        // To fix: keep 'pos' and 'token' outside the loop.

        int pos = 0;
        int token = 128000; // BOS

        while (true)
        {
            Console.Write("User: ");
            var input = Console.ReadLine();
            if (input == "exit") break;

            // Format: <|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
            string formatted = $"<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
            var userTokens = tokeniser.Encode(formatted, false, false);

            foreach (var ut in userTokens)
            {
                transformer.Forward(token, pos++);
                token = ut;
            }

            Console.Write("Assistant: ");
            while (pos < transformer.Config.seq_len)
            {
                var logits = transformer.Forward(token, pos);
                int next = s.Sample(logits);
                pos++;

                if (next == 128009 || next == 128001) // EOT or EOS
                {
                    token = next;
                    break;
                }

                Console.Write(tokeniser.Decode(next));
                token = next;
            }
            Console.WriteLine();
        }
    }
}