using System.Text;

namespace Llama.Models.Llama3;

public class Llama3Tokeniser
{
    public string[] vocab;
    public float[] vocab_scores;
    public int vocab_size;
    public int max_token_length;
    public Dictionary<string, int> sorted_vocab; // Used for fast lookup

    public Llama3Tokeniser(string path, int size)
    {
        vocab_size = size;
        vocab = new string[size];
        vocab_scores = new float[size];
        sorted_vocab = [];

        using var fs = new FileStream(path, FileMode.Open);
        using var br = new BinaryReader(fs);

        max_token_length = br.ReadInt32();
        for (int i = 0; i < size; i++)
        {
            vocab_scores[i] = br.ReadSingle();
            int len = br.ReadInt32();
            byte[] bytes = br.ReadBytes(len);
            // raw bytes -> string (Llama 3 uses byte-level BPE, so we keep 1:1 mapping where possible)
            // For simplicity in C#, we treat the vocab as ISO-8859-1 (raw bytes) or UTF8 depending on impl.
            // Standard Llama 3 is UTF-8 based.
            vocab[i] = Encoding.UTF8.GetString(bytes);

            // Add to lookup if unique (handled lazily in C, here we build dict)
            // Note: Binary vocab might contain duplicates for different merge levels? 
            // C code sorts valid keys. We'll just try/catch add.
            if (!sorted_vocab.ContainsKey(vocab[i]))
                sorted_vocab[vocab[i]] = i;
        }
    }

    public string Decode(int token)
    {
        if (token < 0 || token >= vocab_size)
        {
            return "";
        }

        var piece = vocab[token];

        // Handle <0xXX> tokens
        if (piece.StartsWith("<0x") && piece.EndsWith(">"))
        {
            try
            {
                byte b = Convert.ToByte(piece.Substring(3, 2), 16);
                return ((char)b).ToString();
            }
            catch { return piece; }
        }
        return piece;
    }

    public int[] Encode(string text, bool bos, bool eos)
    {
        var tokens = new List<int>();
        if (bos)
        {
            tokens.Add(128000); // BOS
        }

        // 1. Initial encode: UTF-8 bytes to tokens
        if (!string.IsNullOrEmpty(text))
        {
            // Simple byte fallback: map characters/bytes to tokens
            // This is a simplification. Real Llama 3 tokenization is complex.
            // We use the C approach: Look up exact string, else byte fallback.

            // NOTE: This is a tricky part to port 1:1 without the exact regex splitting logic 
            // used in Python. The C code does a greedy char-by-char or byte-by-byte accumulation.

            byte[] bytes = Encoding.UTF8.GetBytes(text);

            // Very naive C-like port: try to match bytes to vocab id
            // In the C code, it accumulates bytes until not a continuation, then looks up.

            for (int i = 0; i < bytes.Length;)
            {
                // Find longest prefix
                // (The C code is actually simpler: it accumulates UTF8 codepoints then looks up)
                // We will implement the exact C logic logic here roughly

                int len = 1;
                // Check for UTF8 continuation bytes (0b10xxxxxx -> 0x80)
                while (i + len < bytes.Length && (bytes[i + len] & 0xC0) == 0x80 && len < 4)
                {
                    len++;
                }

                string str_buffer = Encoding.UTF8.GetString(bytes, i, len);
                if (sorted_vocab.TryGetValue(str_buffer, out int id))
                {
                    tokens.Add(id);
                }
                else
                {
                    // Byte fallback (+3 shift usually)
                    for (int j = 0; j < len; j++)
                    {
                        tokens.Add(bytes[i + j] + 3);
                    }
                }
                i += len;
            }
        }

        // 2. Merge loop (BPE)
        while (true)
        {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;
            int best_len = 2; // pair or triple

            // Try pairs
            for (int i = 0; i < tokens.Count - 1; i++)
            {
                string s = vocab[tokens[i]] + vocab[tokens[i + 1]];
                if (sorted_vocab.TryGetValue(s, out int id))
                {
                    if (vocab_scores[id] > best_score)
                    {
                        best_score = vocab_scores[id];
                        best_id = id;
                        best_idx = i;
                        best_len = 2;
                    }
                }
            }

            // Optional: Try triples (C code does this)
            // Llama 3 tokenizer sometimes merges 3?
            if (best_idx == -1) // Only if no pair found? C code logic is sequential
            {
                for (int i = 0; i < tokens.Count - 2; i++)
                {
                    string s = vocab[tokens[i]] + vocab[tokens[i + 1]] + vocab[tokens[i + 2]];
                    if (sorted_vocab.TryGetValue(s, out int id))
                    {
                        if (vocab_scores[id] > best_score)
                        {
                            best_score = vocab_scores[id];
                            best_id = id;
                            best_idx = i;
                            best_len = 3;
                        }
                    }
                }
            }

            if (best_idx == -1) break;

            // Merge
            tokens[best_idx] = best_id;
            tokens.RemoveRange(best_idx + 1, best_len - 1);
        }

        if (eos) tokens.Add(128001); // EOS
        return tokens.ToArray();
    }
}
