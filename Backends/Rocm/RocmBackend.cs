using Llama.Backends.Cpu;

namespace Llama.Backends.Rocm;

public unsafe class RocmBackend : IBackend
{
    private readonly IBackend cpuBackend = new CpuBackend();
    private readonly nint rocBlasHandle;

    public RocmBackend()
    {
        RocBlasNative.Check(RocBlasNative.CreateHandle(out rocBlasHandle));
    }

    public void Dispose()
    {
        RocBlasNative.Check(RocBlasNative.DestroyHandle(rocBlasHandle));
    }

    public string GetDescription()
    {
        try
        {
            HipNative.Check(HipNative.GetDeviceCount(out int count));
            HipNative.Check(HipNative.GetDevice(out int currentDev));
            var deviceName = HipNative.GetDeviceName(currentDev);

            return string.Join(Environment.NewLine, Enumerable.Range(0, count).Select(deviceId =>
            {
                var deviceName = HipNative.GetDeviceName(deviceId);
                return $"- GPU {deviceId}: {deviceName}";
            }));
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Failed to query GPU: {ex.Message}");
        }

        return "Unknown GPU";
    }

    public T* Allocate<T>(ulong size) where T : unmanaged
    {
        HipNative.Check(HipNative.HostMalloc(out IntPtr ptr, size, HipNative.HipHostMallocMapped));
        return (T*)ptr;
    }

    public void Free(void* ptr)
    {
        HipNative.Check(HipNative.HostFree(ptr));
    }

    public void MatMul(float* xout, float* x, float* w, int n, int d)
    {
        if (x == null || w == null || xout == null)
        {
            throw new Exception("FATAL: Null pointer passed to GPU!");
        }

        float alpha = 1.0f;
        float beta = 0.0f;

        RocBlasNative.Check(RocBlasNative.Sgemv(
            rocBlasHandle,
            RocBlasNative.Operation.Transpose,
            n, d,
            &alpha,
            w, n,
            x, 1,
            &beta,
            xout, 1));

        HipNative.Check(HipNative.DeviceSynchronize());
    }

    public void RmsNorm(float* o, float* x, float* weight, int size)
    {
        cpuBackend.RmsNorm(o, x, weight, size);
    }

    public void Softmax(float* x, int size)
    {
        cpuBackend.Softmax(x, size);
    }

    public int SampleArgmax(float* probabilities, int n)
    {
        return cpuBackend.SampleArgmax(probabilities, n);
    }

    public int SampleMult(float* probabilities, int n, float coin)
    {
        return cpuBackend.SampleMult(probabilities, n, coin);
    }

    public int SampleTopp(float* probabilities, int n, float topp, ProbIndex[] probindex, float coin)
    {
        return cpuBackend.SampleTopp(probabilities, n, topp, probindex, coin);
    }

    public float RandomF32(ref ulong state)
    {
        return cpuBackend.RandomF32(ref state);
    }
}