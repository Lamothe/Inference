namespace Llama;

public class Tokeniser
{
    public byte[][]? vocab;
    public float[]? vocab_scores;
    public Dictionary<string, int>? sorted_vocab; // Using Dict for O(1) lookup
    public int vocab_size;
    public int max_token_length;
    public byte[][] byte_pieces;

    public Tokeniser(string tokenizerPath, int vocabSize)
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
}
