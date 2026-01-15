using Llama.Backends;

namespace Llama.Models.Llama3;

public class Llama3Sampler(int vocabSize, float temperature, float topp, ulong seed)
{
    public ProbIndex[] probindex = new ProbIndex[vocabSize];

    public unsafe int Sample(float* logits, IBackend backend)
    {
        // Temperature
        if (temperature == 0.0f)
        {
            // Argmax
            int max_i = 0;
            float max_p = logits[0];
            for (int i = 1; i < vocabSize; i++)
            {
                if (logits[i] > max_p) { max_i = i; max_p = logits[i]; }
            }
            return max_i;
        }

        // Apply temperature
        for (int i = 0; i < vocabSize; i++) logits[i] /= temperature;

        // Softmax
        float max_val = logits[0];
        for (int i = 1; i < vocabSize; i++)
        {
            if (logits[i] > max_val) max_val = logits[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < vocabSize; i++)
        {
            logits[i] = MathF.Exp(logits[i] - max_val);
            sum += logits[i];
        }

        for (int i = 0; i < vocabSize; i++)
        {
            logits[i] /= sum;
        }

        // 2. Sampling
        float coin = backend.RandomF32(ref seed);

        if (topp <= 0 || topp >= 1)
        {
            // Standard multinomial
            float cdf = 0.0f;
            for (int i = 0; i < vocabSize; i++)
            {
                cdf += logits[i];
                if (coin < cdf) return i;
            }
            return vocabSize - 1;
        }
        else
        {
            // Top-P (Nucleus)
            var probs = new List<(float prob, int index)>(vocabSize);
            float cutoff = (1.0f - topp) / (vocabSize - 1);

            for (int i = 0; i < vocabSize; i++)
            {
                if (logits[i] >= cutoff) probs.Add((logits[i], i));
            }
            probs.Sort((a, b) => b.prob.CompareTo(a.prob)); // Descending

            float cum_prob = 0.0f;
            int last_idx = probs.Count - 1;
            for (int i = 0; i < probs.Count; i++)
            {
                cum_prob += probs[i].prob;
                if (cum_prob > topp)
                {
                    last_idx = i;
                    break;
                }
            }

            float r = coin * cum_prob;
            float cdf = 0.0f;
            for (int i = 0; i <= last_idx; i++)
            {
                cdf += probs[i].prob;
                if (r < cdf) return probs[i].index;
            }
            return probs[last_idx].index;
        }
    }
}
