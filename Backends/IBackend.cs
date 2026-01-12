namespace Llama.Backends;

public unsafe interface IBackend : IDisposable
{
    string GetDescription();

    T* Allocate<T>(ulong size) where T : unmanaged;

    void Free(void* t);

    void MatMul(float* xout, float* x, float* w, int n, int d);

    void RmsNorm(float* o, float* x, float* weight, int size);

    void Softmax(float* x, int size);

    int SampleArgmax(float* probabilities, int n);

    int SampleMult(float* probabilities, int n, float coin);

    int SampleTopp(float* probabilities, int n, float topp, ProbIndex[] probindex, float coin);

    float RandomF32(ref ulong state);

    int Sample(Sampler s, float* logits);
}
