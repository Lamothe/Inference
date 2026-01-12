# Llama.cs

This is a C# (.NET 10) port of [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy.

Supports selectable backends:
* CPU: Runs purely on the CPU
* ROCm: Uses rocBLAS

## Setup

### Fedora 43

```bash
# Install the .NET 10 SDK
dnf install dotnet-sdk-10.0 g++ rocblas-devel make -y

# Download the small model
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

# OR

# Download the large model
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```

## Run

### Generate
The default run mode (generate) will generate a story using the CPU backend.

```bash
dotnet run
```

Run using the ROCm backend.

```bash
dotnet run -b ROCm
```

### Chat

You can also chat with the models.

```bash
dotnet run -m chat
```
