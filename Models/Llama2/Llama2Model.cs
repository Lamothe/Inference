using System.Globalization;
using System.Numerics.Tensors;
using System.Text;
using Llama.Backends;

namespace Llama.Models.Llama2;

public unsafe class Llama2Model
{
    public static void SafePrint(string piece)
    {
        if (string.IsNullOrEmpty(piece))
        {
            return;
        }

        // Basic check for control chars similar to the C code
        if (piece.Length == 1)
        {
            char c = piece[0];
            if (char.IsControl(c) && !char.IsWhiteSpace(c))
            {
                return;
            }
        }
        Console.Write(piece);
    }

    public int Generate(Llama2Transformer transformer, LLama2Tokeniser tokeniser, Llama2Sampler sampler, string prompt, int steps)
    {
        prompt ??= "";
        var promptTokens = new int[prompt.Length + 3];
        tokeniser.Encode(tokeniser, prompt, true, false, promptTokens, out int numberOfPromptTokens);

        if (numberOfPromptTokens < 1) return;

        long start = 0;
        int next;
        int token = promptTokens[0];
        int tokenNumber = 0;

        while (tokenNumber < steps)
        {
            float* logits = Forward(transformer, token, tokenNumber);

            if (tokenNumber < numberOfPromptTokens - 1)
            {
                next = promptTokens[tokenNumber + 1];
            }
            else
            {
                next = backend.Sample(sampler, logits);
            }
            tokenNumber++;

            if (next == 1)
            {
                break;
            }

            var piece = tokeniser.Decode(tokeniser, token, next);
            SafePrint(piece);
            token = next;
        }
        Console.WriteLine();

        return tokenNumber;
    }

    public void Chat(Llama2Transformer transformer, LLama2Tokeniser tokeniser, Llama2Sampler sampler, string? cli_user, string? cli_sys, int steps)
    {
        string system_prompt = "";
        string user_prompt = "";
        int[] prompt_tokens = new int[2048];
        int num_prompt_tokens = 0;
        int user_idx = 0;
        bool user_turn = true;
        int next = 0;
        int token = 0;
        int pos = 0;

        while (pos < steps)
        {
            if (user_turn)
            {
                if (pos == 0)
                {
                    if (cli_sys == null)
                    {
                        Console.Write("Enter system prompt (optional): ");
                        var input = Console.ReadLine();
                        if (input != null)
                        {
                            system_prompt = input;
                        }
                    }
                    else
                    {
                        system_prompt = cli_sys;
                    }
                }

                if (pos == 0 && cli_user != null)
                {
                    user_prompt = cli_user;
                }
                else
                {
                    Console.Write("User: ");
                    var input = Console.ReadLine();
                    user_prompt = input ?? "";
                }

                string rendered_prompt;
                if (pos == 0 && !string.IsNullOrEmpty(system_prompt))
                {
                    rendered_prompt = $"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]";
                }
                else
                {
                    rendered_prompt = $"[INST] {user_prompt} [/INST]";
                }

                tokeniser.Encode(tokeniser, rendered_prompt, true, false, prompt_tokens, out num_prompt_tokens);
                user_idx = 0;
                user_turn = false;
                Console.Write("Assistant: ");
            }

            if (user_idx < num_prompt_tokens)
            {
                token = prompt_tokens[user_idx++];
            }
            else
            {
                token = next;
            }

            if (token == 2)
            {
                user_turn = true;
            }

            float* logits = Forward(transformer, token, pos);
            next = sampler.Sample(sampler, logits);
            pos++;

            if (user_idx >= num_prompt_tokens && next != 2)
            {
                var piece = tokeniser.Decode(tokeniser, token, next);
                SafePrint(piece);
            }

            if (next == 2)
            {
                Console.WriteLine();
            }
        }
        Console.WriteLine();
    }
}