using System.Globalization;
using System.Numerics.Tensors;
using System.Text;
using Llama.Backends;

namespace Llama;

public unsafe class Engine(IBackend backend)
{
    private static long TimeInMs() => DateTimeOffset.Now.ToUnixTimeMilliseconds();

    public float* Forward(Transformer t, int token, int pos)
    {
        Config p = t.config;
        TransformerWeights w = t.weights;
        RunState s = t.state;
        float* x = s.x;
        int dim = p.dim;
        int kv_dim = p.dim * p.n_kv_heads / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads;
        int hidden_dim = p.hidden_dim;
        int head_size = dim / p.n_heads;

        // copy embedding
        float* content_row = w.token_embedding_table + token * dim;
        Buffer.MemoryCopy(content_row, x, dim * sizeof(float), dim * sizeof(float));

        for (int l = 0; l < p.n_layers; l++)
        {
            // attention rmsnorm
            backend.RmsNorm(s.xb, x, w.rms_att_weight + l * dim, dim);

            // Calculate offsets using the outer 'kv_dim'
            long loff = (long)l * p.seq_len * kv_dim;
            long currentOffset = loff + (long)pos * kv_dim;

            // SAFETY CHECK (Calculate limit locally)
            //    Total floats = Layers * SeqLen * KVDim
            long totalCacheElements = (long)p.n_layers * p.seq_len * kv_dim;

            if (currentOffset + kv_dim > totalCacheElements)
            {
                Console.WriteLine($"CRITICAL: KV Cache Overflow at Layer {l}, Pos {pos}!");
                return null; // Stop safely
            }

            // Assign the pointers
            s.k = s.keyCache + currentOffset;
            s.v = s.valueCache + currentOffset;

            // qkv matmuls
            backend.MatMul(s.q, s.xb, w.wq + (long)l * dim * dim, dim, dim);
            backend.MatMul(s.k, s.xb, w.wk + (long)l * dim * kv_dim, dim, kv_dim);
            backend.MatMul(s.v, s.xb, w.wv + (long)l * dim * kv_dim, dim, kv_dim);

            // RoPE
            for (int i = 0; i < dim; i += 2)
            {
                // Fast lookup, no division or modulo!
                float fcr = MathF.Cos(pos * s.rope_freq[i / 2]);
                float fci = MathF.Sin(pos * s.rope_freq[i / 2]);

                int rotn = i < kv_dim ? 2 : 1; // Existing logic
                for (int v = 0; v < rotn; v++)
                {
                    float* vec = v == 0 ? s.q : s.k;
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // Multihead attention
            Parallel.For(0, p.n_heads, h =>
            {
                float* q = s.q + h * head_size;
                float* att = s.att + h * p.seq_len;
                for (int t_step = 0; t_step <= pos; t_step++)
                {
                    float* k = s.keyCache + loff + t_step * kv_dim + (h / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++)
                    {
                        score += q[i] * k[i];
                    }
                    score /= MathF.Sqrt(head_size);
                    att[t_step] = score;
                }

                backend.Softmax(att, pos + 1);

                float* xb = s.xb + h * head_size;
                // memset 0
                new Span<float>(xb, head_size).Clear();

                for (int t_step = 0; t_step <= pos; t_step++)
                {
                    float* v = s.valueCache + loff + t_step * kv_dim + (h / kv_mul) * head_size;
                    float a = att[t_step];
                    for (int i = 0; i < head_size; i++)
                    {
                        xb[i] += a * v[i];
                    }
                }
            });

            // final matmul
            backend.MatMul(s.xb2, s.xb, w.wo + (long)l * dim * dim, dim, dim);

            // residual connection back into x
            var x_span = new Span<float>(x, dim);
            var xb2_span = new ReadOnlySpan<float>(s.xb2, dim);

            // In-place addition: x = x + xb2
            TensorPrimitives.Add(x_span, xb2_span, x_span);

            // ffn rmsnorm
            backend.RmsNorm(s.xb, x, w.rms_ffn_weight + l * dim, dim);

            // ffn
            backend.MatMul(s.hb, s.xb, w.w1 + (long)l * dim * hidden_dim, dim, hidden_dim);
            backend.MatMul(s.hb2, s.xb, w.w3 + (long)l * dim * hidden_dim, dim, hidden_dim);

            // SwiGLU
            Parallel.For(0, hidden_dim, i =>
            {
                float val = s.hb[i];
                val *= 1.0f / (1.0f + MathF.Exp(-val));
                val *= s.hb2[i];
                s.hb[i] = val;
            });

            backend.MatMul(s.xb, s.hb, w.w2 + (long)l * dim * hidden_dim, hidden_dim, dim);

            // residual
            for (int i = 0; i < dim; i++) x[i] += s.xb[i];
        }

        backend.RmsNorm(x, x, w.rms_final_weight, dim);
        backend.MatMul(s.logits, x, w.wcls, p.dim, p.vocab_size);
        return s.logits;
    }

    public static string Decode(Tokeniser t, int prev_token, int token)
    {
        byte[] piece = t.vocab![token];
        // strip leading whitespace if prev was BOS (1)
        if (prev_token == 1 && piece.Length > 0 && piece[0] == ' ')
        {
            piece = piece[1..];
        }

        // Check for raw byte tokens like <0x01>
        var pieceStr = Encoding.UTF8.GetString(piece);
        if (pieceStr.StartsWith("<0x") && pieceStr.EndsWith(">"))
        {
            if (byte.TryParse(pieceStr.AsSpan(3, 2), NumberStyles.HexNumber, null, out byte byteVal))
            {
                return Encoding.Latin1.GetString(new byte[] { byteVal });
            }
        }
        return pieceStr;
    }

    public static void SafePrint(string piece)
    {
        if (string.IsNullOrEmpty(piece)) return;
        // Basic check for control chars similar to the C code
        if (piece.Length == 1)
        {
            char c = piece[0];
            if (char.IsControl(c) && !char.IsWhiteSpace(c)) return;
        }
        Console.Write(piece);
    }

    public static int StrLookup(string str, Tokeniser t)
    {
        return t.sorted_vocab!.TryGetValue(str, out int id) ? id : -1;
    }

    public static void Encode(Tokeniser t, string text, bool bos, bool eos, int[] tokens, out int n_tokens)
    {
        if (text == null)
        {
            n_tokens = 0; return;
        }

        var tokenList = new List<int>();
        if (bos)
        {
            tokenList.Add(1);
        }

        if (!string.IsNullOrEmpty(text))
        {
            int dummy_prefix = StrLookup(" ", t);
            tokenList.Add(dummy_prefix);
        }

        // UTF-8 byte processing
        byte[] textBytes = Encoding.UTF8.GetBytes(text);

        for (int i = 0; i < textBytes.Length; i++)
        {
            // Try to find the longest matching token from byte stream 
            // The C code actually looks up by codepoint (parsing UTF8 chars), then falls back to bytes.
            // Simplified here: Get UTF8 char (rune)

            // Note: The C code logic for breaking down UTF8 to codepoints is complex to map 1:1 exactly 
            // without unsafe pointer arithmetic on the string.
            // We will mimic the behavior: Read one rune, try lookup, else byte fallback.

            // Find length of current UTF8 sequence
            int len = 1;
            if ((textBytes[i] & 0xC0) != 0x80) // Start of char
            {
                if ((textBytes[i] & 0xE0) == 0xC0)
                {
                    len = 2;
                }
                else if ((textBytes[i] & 0xF0) == 0xE0)
                {
                    len = 3;
                }
                else if ((textBytes[i] & 0xF8) == 0xF0)
                {
                    len = 4;
                }
            }

            // Check boundary
            if (i + len > textBytes.Length) len = 1; // Should not happen in valid UTF8

            var substr = new byte[len];
            Array.Copy(textBytes, i, substr, 0, len);

            var strKey = Encoding.UTF8.GetString(substr);
            var id = StrLookup(strKey, t);

            if (id != -1)
            {
                tokenList.Add(id);
            }
            else
            {
                // Byte fallback
                for (int j = 0; j < len; j++)
                {
                    tokenList.Add(textBytes[i + j] + 3);
                }
            }
            i += (len - 1);
        }

        // BPE Merge
        while (true)
        {
            var best_score = -1e10f;
            var best_id = -1;
            var best_idx = -1;

            for (int i = 0; i < tokenList.Count - 1; i++)
            {
                // Reconstruct string for pair
                var b1 = t.vocab![tokenList[i]];
                var b2 = t.vocab[tokenList[i + 1]];

                // Concatenate bytes
                var merged = new byte[b1.Length + b2.Length];
                b1.CopyTo(merged, 0);
                b2.CopyTo(merged, b1.Length);

                var mergeStr = Encoding.UTF8.GetString(merged);

                var id = StrLookup(mergeStr, t);
                if (id != -1 && t.vocab_scores![id] > best_score)
                {
                    best_score = t.vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1)
            {
                break;
            }

            tokenList[best_idx] = best_id;
            tokenList.RemoveAt(best_idx + 1);
        }

        if (eos)
        {
            tokenList.Add(2);
        }

        n_tokens = tokenList.Count;
        for (int i = 0; i < n_tokens; i++)
        {
            tokens[i] = tokenList[i];
        }
    }

    public void Generate(Transformer t, Tokeniser tokeniser, Sampler sampler, string prompt, int steps)
    {
        prompt ??= "";
        int[] prompt_tokens = new int[prompt.Length + 3];
        int num_prompt_tokens;
        Encode(tokeniser, prompt, true, false, prompt_tokens, out num_prompt_tokens);

        if (num_prompt_tokens < 1) return;

        long start = 0;
        int next;
        int token = prompt_tokens[0];
        int pos = 0;

        while (pos < steps)
        {
            float* logits = Forward(t, token, pos);

            if (pos < num_prompt_tokens - 1)
            {
                next = prompt_tokens[pos + 1];
            }
            else
            {
                next = backend.Sample(sampler, logits);
            }
            pos++;

            if (next == 1) break;

            string piece = Decode(tokeniser, token, next);
            SafePrint(piece);
            token = next;

            if (start == 0) start = TimeInMs();
        }
        Console.WriteLine();

        if (pos > 1)
        {
            long end = TimeInMs();
            Console.Error.WriteLine($"Token/s: {(pos - 1) / (double)(end - start) * 1000}");
        }
    }

    public void Chat(Transformer t, Tokeniser tokeniser, Sampler sampler, string? cli_user, string? cli_sys, int steps)
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
                        string? input = Console.ReadLine();
                        if (input != null) system_prompt = input;
                    }
                    else system_prompt = cli_sys;
                }

                if (pos == 0 && cli_user != null)
                {
                    user_prompt = cli_user;
                }
                else
                {
                    Console.Write("User: ");
                    string? input = Console.ReadLine();
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

                Encode(tokeniser, rendered_prompt, true, false, prompt_tokens, out num_prompt_tokens);
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

            if (token == 2) user_turn = true;

            float* logits = Forward(t, token, pos);
            next = backend.Sample(sampler, logits);
            pos++;

            if (user_idx >= num_prompt_tokens && next != 2)
            {
                string piece = Decode(tokeniser, token, next);
                SafePrint(piece);
            }
            if (next == 2) Console.WriteLine();
        }
        Console.WriteLine();
    }
}