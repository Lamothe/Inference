namespace Llama2;

public class Tokenizer
{
    public byte[][]? vocab;
    public float[]? vocab_scores;
    public Dictionary<string, int>? sorted_vocab; // Using Dict for O(1) lookup
    public int vocab_size;
    public int max_token_length;
    public byte[][] byte_pieces;

    public Tokenizer()
    {
        byte_pieces = new byte[256][];
        for (int i = 0; i < 256; i++)
        {
            byte_pieces[i] = [(byte)i];
        }
    }
}
