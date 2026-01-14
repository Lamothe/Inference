using System.Globalization;
using System.Text;

namespace Llama.Models.Llama2;

public class LLama2Tokeniser
{
    public byte[][]? vocab;
    public float[]? vocab_scores;
    public Dictionary<string, int>? sorted_vocab; // Using Dict for O(1) lookup
    public int vocab_size;
    public int max_token_length;
    public byte[][] byte_pieces;

    public LLama2Tokeniser(string tokenizerPath, int vocabSize)
    {
        byte_pieces = new byte[256][];
        for (int i = 0; i < 256; i++)
        {
            byte_pieces[i] = [(byte)i];
        }

        vocab_size = vocabSize;
        vocab = new byte[vocab_size][];
        vocab_scores = new float[vocab_size];
        sorted_vocab = new Dictionary<string, int>();

        using var fs = new FileStream(tokenizerPath, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);

        max_token_length = br.ReadInt32();

        for (int i = 0; i < vocab_size; i++)
        {
            vocab_scores[i] = br.ReadSingle();
            var len = br.ReadInt32();
            vocab[i] = br.ReadBytes(len); // Read exact bytes, no encoding assumption yet

            // For dictionary lookup, we convert to string (Latin1 to preserve bytes 1:1 if needed, 
            // but usually UTF8 is intended here). C code uses raw bytes -> strcmp.
            // Using UTF8 for the dictionary key is safer for standard tex
            var str = System.Text.Encoding.UTF8.GetString(vocab[i]);
            sorted_vocab[str] = i;
        }
    }

    public int StrLookup(string str)
    {
        return sorted_vocab!.TryGetValue(str, out int id) ? id : -1;
    }

    public string Decode(int prev_token, int token)
    {
        byte[] piece = vocab![token];
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

    public void Encode(string text, bool bos, bool eos, int[] tokens, out int n_tokens)
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
            int dummyPrefix = StrLookup(" ");
            tokenList.Add(dummyPrefix);
        }

        // UTF-8 byte processing
        var textBytes = Encoding.UTF8.GetBytes(text);

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
            var id = StrLookup(strKey);

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
                var b1 = vocab![tokenList[i]];
                var b2 = vocab[tokenList[i + 1]];

                // Concatenate bytes
                var merged = new byte[b1.Length + b2.Length];
                b1.CopyTo(merged, 0);
                b2.CopyTo(merged, b1.Length);

                var mergeStr = Encoding.UTF8.GetString(merged);

                var id = StrLookup(mergeStr);
                if (id != -1 && vocab_scores![id] > best_score)
                {
                    best_score = vocab_scores[id];
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
        for (var i = 0; i < n_tokens; i++)
        {
            tokens[i] = tokenList[i];
        }
    }
}
