using LLamaSharp.KernelMemory;
using Microsoft.KernelMemory;
using Microsoft.KernelMemory.Configuration;
using Microsoft.KernelMemory.ContentStorage.DevTools;
using Microsoft.KernelMemory.FileSystem.DevTools;
using Microsoft.KernelMemory.MemoryStorage.DevTools;
using System.Diagnostics;
using LLama.Common;
using Microsoft.Identity.Client;
using System.Threading;
namespace LLama.Examples.Examples;

public class KernelMemorySaveAndLoad
{
    static string StorageFolder => Path.GetFullPath($"./storage-{nameof(KernelMemorySaveAndLoad)}");
    static bool StorageExists => Directory.Exists(StorageFolder) && Directory.GetDirectories(StorageFolder).Length > 0;

    LlamaSharpTextGenerator _generator = null;
    public async Task Run()
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine(
            """

            This program uses the Microsoft.KernelMemory package to ingest documents
            and store the embeddings as local files so they can be quickly recalled
            when this application is launched again. 

            """);

        string modelPath = UserSettings.GetModelPath();
        IKernelMemory memory = CreateMemoryWithLocalStorage(modelPath);

        Console.ForegroundColor = ConsoleColor.Yellow;
        if (StorageExists)
        {
            Console.WriteLine(
                """
                
                Kernel memory files have been located!
                Information about previously analyzed documents has been loaded.

                """);
        }
        else
        {
            Console.WriteLine(
                $"""

                 Existing kernel memory was not found.
                 Documents will be analyzed (slow) and information saved to disk.
                 Analysis will not be required the next time this program is run.
                 Press ENTER to proceed...
 
                 """);
            Console.ReadLine();
            await IngestDocuments(memory);
        }

        await AskSingleQuestion(memory, "What are the required documents?");
        await StartUserChatSession(memory);
    }

    private IKernelMemory CreateMemoryWithLocalStorage(string modelPath)
    {
        Common.InferenceParams infParams = new() { AntiPrompts = ["\n\n"] };

        LLamaSharpConfig lsConfig = new(modelPath) { DefaultInferenceParams = infParams };

        SearchClientConfig searchClientConfig = new()
        {
            MaxMatchesCount = 2,
            AnswerTokens = 100,
        };

        TextPartitioningOptions parseOptions = new()
        {
            MaxTokensPerParagraph = 200,
            MaxTokensPerLine = 70,
            OverlappingTokens = 25
        };

        SimpleFileStorageConfig storageConfig = new()
        {
            Directory = StorageFolder,
            StorageType = FileSystemTypes.Disk,
        };

        SimpleVectorDbConfig vectorDbConfig = new()
        {
            Directory = StorageFolder,
            StorageType = FileSystemTypes.Disk,
        };

        // Create the executor object here
        var parameters = new ModelParams(lsConfig.ModelPath)
        {
            ContextSize = lsConfig.ContextSize ?? 2048,
            Seed = lsConfig.Seed ?? 0,
            GpuLayerCount = lsConfig.GpuLayerCount ?? 20,
            Embeddings = true,
            MainGpu = lsConfig?.MainGpu ?? 0,
            SplitMode = lsConfig?.SplitMode ?? LLama.Native.GPUSplitMode.None,
        };
        var weights = LLamaWeights.LoadFromFile(parameters);
        var context = weights.CreateContext(parameters);
        var executor = new StatelessExecutor(weights, parameters);
        _generator = new LlamaSharpTextGenerator(weights, context, executor, lsConfig.DefaultInferenceParams);
        var embedder = new LLamaEmbedder(weights, parameters);

        Console.ForegroundColor = ConsoleColor.Blue;
        Console.WriteLine($"Kernel memory folder: {StorageFolder}");

        Console.ForegroundColor = ConsoleColor.DarkGray;
        return new KernelMemoryBuilder()
            .WithSimpleFileStorage(storageConfig)
            .WithSimpleVectorDb(vectorDbConfig)
            .WithLLamaSharpCustom(lsConfig, _generator, embedder) // Pass LlamaSharpTextGenerator and LLamaEmbedder instances
            .WithSearchClientConfig(searchClientConfig)
            .With(parseOptions)
            .Build();
    }

    private async Task AskSingleQuestion(IKernelMemory memory, string question)
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"Question: {question}");
        await ShowAnswer(memory, question);
    }

    private async Task StartUserChatSession(IKernelMemory memory)
    {
        while (true)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write("Question: ");
            string question = Console.ReadLine()!;
            if (string.IsNullOrEmpty(question))
                return;

            await ShowAnswer(memory, question);
        }
    }

    private async Task IngestDocuments(IKernelMemory memory)
    {
        string[] filesToIngest = [
                Path.GetFullPath(@"./Assets/Admissions IIM A.pdf"),
               // Path.GetFullPath(@"./Assets/sample-KM-Readme.pdf"),
            ];

        for (int i = 0; i < filesToIngest.Length; i++)
        {
            string path = filesToIngest[i];
            Stopwatch sw = Stopwatch.StartNew();
            Console.ForegroundColor = ConsoleColor.Blue;
            Console.WriteLine($"Importing {i + 1} of {filesToIngest.Length}: {path}");
            await memory.ImportDocumentAsync(path, steps: Constants.PipelineWithoutSummary);
            Console.WriteLine($"Completed in {sw.Elapsed}\n");
        }
    }

    private async Task ShowAnswer(IKernelMemory memory, string question)
    {
        Stopwatch sw = Stopwatch.StartNew();
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"Generating answer...");
        MemoryAnswer answer = await memory.AskAsync(question);

        // given the facts below
        var generatingText = _generator.GenerateLiveTextAsync(answer.Result);
        
        // Display the generating text async
        await foreach (var text in generatingText)
        {
            Console.ForegroundColor = ConsoleColor.Gray;
            // print the text not in new line
            Console.Write(text);
        }
        Console.ForegroundColor = ConsoleColor.Gray;
        Console.WriteLine($"Answer: {answer.Result}");
        foreach (var source in answer.RelevantSources)
        {
            Console.WriteLine($"Source: {source.SourceName}");
        }
        Console.WriteLine($"Answer generated in {sw.Elapsed}");

        Console.WriteLine();
    }
}