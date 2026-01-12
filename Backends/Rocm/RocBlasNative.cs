using System.Runtime.InteropServices;

namespace Llama.Backends.Rocm;

// Note: rocBLAS expects pointers to GPU memory, not managed arrays!
public unsafe static class RocBlasNative
{
    private const string LibName = "librocblas.so";

    public enum Status : int
    {
        Success = 0,
        InvalidHandle = 1,
        NotImplemented = 2,
        InvalidPointer = 3,
        InvalidSize = 4,
        MemoryError = 5,
        InternalError = 6,
        PerfDegraded = 7,
        SizeQueryMismatch = 8,
        SizeIncreased = 9,
        SizeUnchanged = 10,
        InvalidValue = 11,
        Continue = 12
    }

    public static void Check(int statusInt)
    {
        var status = (Status)statusInt;
        if (status != Status.Success)
        {
            string message = status switch
            {
                Status.InvalidHandle => "Handle not initialized, invalid or null",
                Status.NotImplemented => "Function not implemented",
                Status.InvalidPointer => "Invalid pointer parameter",
                Status.InvalidSize => "Invalid size parameter",
                Status.MemoryError => "Failed memory allocation, copy, or dealloc",
                Status.InternalError => "Internal library failure",
                Status.PerfDegraded => "Performance degraded due to low device memory",
                Status.InvalidValue => "Invalid value parameter",
                _ => $"Unknown rocBLAS Error: {status}"
            };

            throw new Exception($"rocBLAS Error: {message} ({status})");
        }
    }

    [DllImport(LibName, EntryPoint = "rocblas_create_handle")]
    public static extern int CreateHandle(out IntPtr handle);

    [DllImport(LibName, EntryPoint = "rocblas_destroy_handle")]
    public static extern int DestroyHandle(IntPtr handle);

    // Wrap the Operation enums for readability
    public enum Operation : int
    {
        None = 111,
        Transpose = 112,
        ConjugateTranspose = 113
    }

    // rocblas_sgemv
    // y = alpha * A * x + beta * y
    [DllImport(LibName, EntryPoint = "rocblas_sgemv")]
    public static extern int Sgemv(
        IntPtr handle,
        Operation trans,
        int m,
        int n,
        float* alpha, // Host pointer
        void* A,      // Device pointer
        int lda,
        void* x,      // Device pointer
        int incx,
        float* beta,  // Host pointer
        void* y,      // Device pointer
        int incy
    );
}
