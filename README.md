# Llama2.cs

This is a C# (.NET 10) port of [llama.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy.
The transformer model runs completely in system memory on the CPU, no GPU/APU support, 100% C#.

## Setup

### Fedora 43

```bash
# Install the .NET 10 SDK
dnf install dotnet-sdk-10.0

# Download the small model
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

## Run

### Generate
The default run mode (generate) will generate a story.

```bash
dotnet run
```

### Chat

You can also chat with the models.

```bash
dotnet run -m chat
```
