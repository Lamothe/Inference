using System.Numerics.Tensors;

namespace Llama2;

public unsafe class Operations
{
    public static void MatMul(float* xout, float* x, float* w, int n, int d)
    {
        // Using Parallel.For to parallelize over the rows (output dimension)
        Parallel.For(0, d, i =>
        {
            // specific row of weights
            // Create spans from pointers (zero-cost abstraction)
            var w_row = new ReadOnlySpan<float>(w + (long)i * n, n);
            var x_vec = new ReadOnlySpan<float>(x, n);

            // TensorPrimitives.Dot automatically uses AVX2/AVX-512/NEON
            xout[i] = TensorPrimitives.Dot(w_row, x_vec);
        });
    }

    public static void RmsNorm(float* o, float* x, float* weight, int size)
    {
        // Create spans
        var x_span = new ReadOnlySpan<float>(x, size);
        var w_span = new ReadOnlySpan<float>(weight, size);
        var o_span = new Span<float>(o, size);

        // 1. Calculate Sum of Squares
        float ss = TensorPrimitives.SumOfSquares(x_span);

        // 2. Normalization constant
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / MathF.Sqrt(ss);

        // Simplest vectorization using TensorPrimitives for the element-wise part:
        // This multiplies x elements by the scalar 'ss' and stores in 'o'
        TensorPrimitives.Multiply(x_span, ss, o_span);
        // Then multiply 'o' by the weights
        TensorPrimitives.Multiply(o_span, w_span, o_span);
    }

    public static void Softmax(float* x, int size)
    {
        float max_val = x[0];
        for (int i = 1; i < size; i++)
        {
            if (x[i] > max_val) max_val = x[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            x[i] = MathF.Exp(x[i] - max_val);
            sum += x[i];
        }
        for (int i = 0; i < size; i++)
        {
            x[i] /= sum;
        }
    }

    public static int SampleArgmax(float* probabilities, int n)
    {
        int max_i = 0;
        float max_p = probabilities[0];
        for (int i = 1; i < n; i++)
        {
            if (probabilities[i] > max_p)
            {
                max_i = i;
                max_p = probabilities[i];
            }
        }
        return max_i;
    }

    public static int SampleMult(float* probabilities, int n, float coin)
    {
        float cdf = 0.0f;
        for (int i = 0; i < n; i++)
        {
            cdf += probabilities[i];
            if (coin < cdf) return i;
        }
        return n - 1;
    }

    public static int SampleTopp(float* probabilities, int n, float topp, ProbIndex[] probindex, float coin)
    {
        int n0 = 0;
        float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < n; i++)
        {
            if (probabilities[i] >= cutoff)
            {
                probindex[n0].index = i;
                probindex[n0].prob = probabilities[i];
                n0++;
            }
        }

        Array.Sort(probindex, 0, n0);

        float cumulative_prob = 0.0f;
        int last_idx = n0 - 1;
        for (int i = 0; i < n0; i++)
        {
            cumulative_prob += probindex[i].prob;
            if (cumulative_prob > topp)
            {
                last_idx = i;
                break;
            }
        }

        float r = coin * cumulative_prob;
        float cdf = 0.0f;
        for (int i = 0; i <= last_idx; i++)
        {
            cdf += probindex[i].prob;
            if (r < cdf) return probindex[i].index;
        }
        return probindex[last_idx].index;
    }

    public static float RandomF32(ref ulong state)
    {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        uint res = (uint)((state * 0x2545F4914F6CDD1Dul) >> 32);
        return (res >> 8) / 16777216.0f;
    }

    public static int Sample(Sampler s, float* logits)
    {
        if (s.temperature == 0.0f) return SampleArgmax(logits, s.vocab_size);

        for (int q = 0; q < s.vocab_size; q++) logits[q] /= s.temperature;
        Softmax(logits, s.vocab_size);

        float coin = RandomF32(ref s.rng_state);

        if (s.topp <= 0 || s.topp >= 1)
        {
            return SampleMult(logits, s.vocab_size, coin);
        }
        else
        {
            return SampleTopp(logits, s.vocab_size, s.topp, s.probindex, coin);
        }
    }
}