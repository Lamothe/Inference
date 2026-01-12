using System.Runtime.InteropServices;

namespace Llama.Backends.Rocm;

public unsafe static class HipNative
{
    private const string LibName = "libamdhip64.so";

    [DllImport(LibName, EntryPoint = "hipGetDeviceCount")]
    public static extern int GetDeviceCount(out int count);

    [DllImport(LibName, EntryPoint = "hipGetDevice")]
    public static extern int GetDevice(out int deviceId);

    // We use void* here to avoid defining the massive, version-unstable hipDeviceProp_t struct
    [DllImport(LibName, EntryPoint = "hipGetDeviceProperties")]
    public static extern int GetDeviceProperties(void* prop, int deviceId);

    /// <summary>
    ///  Helper to safely read the device name without a struct definition.
    /// </summary>
    public static string GetDeviceName(int deviceId)
    {
        // Allocate a buffer large enough for the struct (2KB is plenty for current versions)
        // We use stackalloc for speed and automatic cleanup
        const int MaxStructSize = 2048;
        byte* buffer = stackalloc byte[MaxStructSize];

        // Zero out memory just to be safe
        new Span<byte>(buffer, MaxStructSize).Clear();

        Check(GetDeviceProperties(buffer, deviceId));

        // The 'name' field is char[256] and is at offset 0
        return Marshal.PtrToStringAnsi((IntPtr)buffer) ?? "Unknown Device";
    }

    // Flags for hipHostMalloc
    public const uint HipHostMallocDefault = 0x00;
    public const uint HipHostMallocPortable = 0x01;
    public const uint HipHostMallocMapped = 0x02;
    public const uint HipHostMallocWriteCombined = 0x04;

    // hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
    [DllImport(LibName, EntryPoint = "hipHostMalloc")]
    public static extern int HostMalloc(out IntPtr ptr, ulong size, uint flags);

    // hipError_t hipHostFree(void* ptr);
    [DllImport(LibName, EntryPoint = "hipHostFree")]
    public static extern int HostFree(void* ptr);

    // Helper to get the device pointer if UVA (Unified Virtual Addressing) isn't fully active
    // hipError_t hipHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int flags);
    [DllImport(LibName, EntryPoint = "hipHostGetDevicePointer")]
    public static extern int HostGetDevicePointer(out void* devPtr, void* hostPtr, uint flags);

    // Synchronize CPU with GPU
    // Critical: Blocks C# thread until GPU finishes all queued tasks
    [DllImport(LibName, EntryPoint = "hipDeviceSynchronize")]
    public static extern int DeviceSynchronize();

    // const char* hipGetErrorString(hipError_t hipError);
    [DllImport(LibName, EntryPoint = "hipGetErrorString")]
    private static extern IntPtr GetErrorStringNative(int error);

    /// <summary>
    /// Checks the HIP error code. If it's not Success (0), throws an exception with the readable error message.
    /// </summary>
    public static void Check(int error)
    {
        if (error != 0) // 0 is usually hipSuccess
        {
            IntPtr ptr = GetErrorStringNative(error);
            string message = Marshal.PtrToStringAnsi(ptr) ?? "Unknown Error";
            throw new Exception($"HIP Error {error}: {message}");
        }
    }
}