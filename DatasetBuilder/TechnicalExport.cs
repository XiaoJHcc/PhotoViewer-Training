using System.Buffers.Binary;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using PhotoViewer.Core.AI;
using PhotoViewer.Core.Database;
using PhotoViewer.Core.Image;

namespace DatasetBuilder;

/// <summary>按固定来源清单导出带身份哈希的修复 CV/patch 侧缓存；不写数据集库或原始照片。</summary>
internal static class TechnicalExport
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    /// <summary>解析技术导出参数并执行真实提取，返回进程退出码。</summary>
    /// <param name="args">来源清单、输出目录、并发与可选 patch/渲染参数。</param>
    /// <returns>成功返回 0，输入或提取异常返回 3。</returns>
    public static int Run(string[] args)
    {
        try
        {
            string? sources = null, output = null, model = null;
            bool patches = false, renders = false;
            int concurrency = 2;
            for (int index = 0; index < args.Length; index++)
            {
                switch (args[index])
                {
                    case "--dump-technical": sources = args[++index]; break;
                    case "--output": output = args[++index]; break;
                    case "--model-file": model = args[++index]; break;
                    case "--concurrency": concurrency = int.Parse(args[++index]); break;
                    case "--patches": patches = true; break;
                    case "--renders": renders = true; break;
                    default: throw new ArgumentException($"未知技术导出参数: {args[index]}");
                }
            }
            if (sources == null || output == null || concurrency < 1 || patches != (model != null))
                throw new ArgumentException("需要 --dump-technical <来源JSON> --output <目录>；--patches 与 --model-file 必须同时提供；并发须大于0。");
            return ExportAsync(sources, output, concurrency, patches, renders, model).GetAwaiter().GetResult();
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"TECHNICAL ERROR: {exception}");
            return 3;
        }
        finally
        {
            DinoFeatureExtractor.UnloadSession();
        }
    }

    /// <summary>核验来源身份并并发导出不可变结果；完整成功后写最终覆盖闸报告。</summary>
    /// <param name="sourcesPath">包含 Fingerprint/Path 的 JSON 数组。</param>
    /// <param name="output">独立缓存目录。</param>
    /// <param name="concurrency">同时解码的文件数。</param>
    /// <param name="patches">是否提取双输出 ONNX。</param>
    /// <param name="renders">是否保存共享预处理 PNG。</param>
    /// <param name="modelPath">patch 模型路径，不提取时为空。</param>
    /// <returns>所有请求完成后返回 0，异常由调用方处理。</returns>
    private static async Task<int> ExportAsync(string sourcesPath, string output, int concurrency,
                                              bool patches, bool renders, string? modelPath)
    {
        byte[] sourceBytes = File.ReadAllBytes(sourcesPath);
        var sources = JsonSerializer.Deserialize<Source[]>(sourceBytes)
            ?? throw new InvalidDataException("技术来源清单为空。");
        if (sources.Length == 0 || sources.Select(source => source.Fingerprint).Distinct().Count() != sources.Length)
            throw new InvalidDataException("技术来源清单为空或指纹重复。");
        foreach (var source in sources)
        {
            if (source.Fingerprint.Length != 40 || source.Fingerprint.Any(character => !"0123456789abcdef".Contains(character)))
                throw new InvalidDataException($"非法指纹: {source.Fingerprint}");
            if (!File.Exists(source.Path) || !PhotoDecode.CanDecode(source.Path))
                throw new InvalidDataException($"缺少可解码原片: {source.Path}");
        }
        var identity = new
        {
            format = "technical-cache-v1",
            decode_version = PhotoDecode.Version,
            cv_spec = $"{CvGridResult.CurrentVersion}+decode-{PhotoDecode.Version}",
            model_id = patches ? $"{DinoModelResources.ModelId}+decode-{PhotoDecode.Version}" : null,
            sources_sha256 = Hash(sourceBytes),
            product_assembly_sha256 = HashFile(typeof(CvGridExtractor).Assembly.Location),
            exporter_assembly_sha256 = HashFile(typeof(TechnicalExport).Assembly.Location),
            model_sha256 = modelPath != null ? HashFile(modelPath) : null,
            scalar_names = CvGridResult.ScalarNames,
            grid_size = CvGridResult.GridSize,
            input_size = DinoModelResources.InputSize,
            patch_size = DinoModelResources.PatchSize,
            save_patches = patches,
            save_renders = renders,
            requested = sources.Length
        };
        Directory.CreateDirectory(output);
        WriteImmutable(Path.Combine(output, "identity.json"), JsonSerializer.SerializeToUtf8Bytes(identity, JsonOptions));
        if (modelPath != null)
            DinoFeatureExtractor.ConfigureModelOverride(File.ReadAllBytes(modelPath));
        var stopwatch = Stopwatch.StartNew();
        int done = 0, reused = 0;
        await Parallel.ForEachAsync(sources, new ParallelOptions { MaxDegreeOfParallelism = concurrency }, async (source, cancellation) =>
        {
            string prefix = Path.Combine(output, source.Fingerprint);
            string recordPath = prefix + ".json";
            string sourceHash = HashFile(source.Path);
            if (File.Exists(recordPath))
            {
                ValidateExisting(recordPath, prefix, source, sourceHash, patches, renders);
                Interlocked.Increment(ref reused);
            }
            else
            {
                var exif = PhotoDecode.ReadExif(source.Path);
                var fingerprint = PhotoFingerprint.Compute(new PhotoFingerprintInput
                {
                    FilenameNoExt = Path.GetFileNameWithoutExtension(source.Path),
                    CaptureTime = exif.CaptureTime ?? File.GetLastWriteTimeUtc(source.Path),
                    CaptureSubSec = exif.SubSec
                });
                if (fingerprint != source.Fingerprint)
                    throw new InvalidDataException($"原片指纹已漂移: {source.Path}");
                using var image = PhotoDecode.LoadBitmap(source.Path)
                    ?? throw new InvalidDataException($"原片解码失败: {source.Path}");
                var cv = await CvGridExtractor.ExtractAsync(image, cancellation);
                if (cv.Data.Length != CvGridResult.DataLength || cv.Data.Any(float.IsInfinity))
                    throw new InvalidDataException("CV 输出形状或有限性错误。");
                byte[] cvBytes = cv.Encode();
                string? clsHash = null, patchHash = null, renderHash = null;
                if (patches)
                {
                    var dual = await DinoFeatureExtractor.ExtractDualAsync(image, includePatches: true, cancellation);
                    if (dual.Cls.Length != DinoModelResources.FeatureDim || dual.Patches?.Length != DinoModelResources.PatchTokenCount * DinoModelResources.FeatureDim)
                        throw new InvalidDataException("ONNX 双输出形状错误。");
                    byte[] clsBytes = EncodeFloats(dual.Cls), patchBytes = EncodeFloats(dual.Patches);
                    WriteImmutable(prefix + ".cls.bin", clsBytes);
                    WriteImmutable(prefix + ".patch.bin", patchBytes);
                    clsHash = Hash(clsBytes);
                    patchHash = Hash(patchBytes);
                }
                if (renders)
                {
                    using var input = DinoFeatureExtractor.RenderInputImage(image);
                    using var stream = new MemoryStream();
                    SkiaImageCodec.SavePng(input, stream);
                    byte[] renderBytes = stream.ToArray();
                    WriteImmutable(prefix + ".png", renderBytes);
                    renderHash = Hash(renderBytes);
                }
                if (HashFile(source.Path) != sourceHash)
                    throw new InvalidDataException($"提取期间原片发生变化: {source.Path}");
                WriteImmutable(prefix + ".cv.bin", cvBytes);
                WriteImmutable(recordPath, JsonSerializer.SerializeToUtf8Bytes(new
                {
                    fingerprint = source.Fingerprint,
                    source_path = Path.GetFullPath(source.Path),
                    source_sha256 = sourceHash,
                    width = image.Width,
                    height = image.Height,
                    cv_sha256 = Hash(cvBytes),
                    cls_sha256 = clsHash,
                    patches_sha256 = patchHash,
                    render_sha256 = renderHash
                }, JsonOptions));
            }
            int completed = Interlocked.Increment(ref done);
            if (completed % 100 == 0 || completed == sources.Length)
                Console.WriteLine($"TECHNICAL [{completed}/{sources.Length}] reused={Volatile.Read(ref reused)} elapsed={stopwatch.Elapsed.TotalSeconds:F1}s");
        });
        if (HashFile(sourcesPath) != identity.sources_sha256)
            throw new InvalidDataException("提取期间来源清单发生变化。");
        WriteImmutable(Path.Combine(output, "complete.json"), JsonSerializer.SerializeToUtf8Bytes(new
        {
            requested = sources.Length,
            completed = done,
            identity_sha256 = HashFile(Path.Combine(output, "identity.json"))
        }, JsonOptions));
        Console.WriteLine($"TECHNICAL GATE: complete={done}/{sources.Length}, reused={reused}, seconds={stopwatch.Elapsed.TotalSeconds:F1}");
        return 0;
    }

    /// <summary>核验已完成缓存的来源及所有要求输出哈希，禁止用残缺或漂移结果续跑。</summary>
    /// <param name="recordPath">单图完成记录路径。</param>
    /// <param name="prefix">指纹文件前缀。</param>
    /// <param name="source">请求来源。</param>
    /// <param name="sourceHash">当前原片哈希。</param>
    /// <param name="patches">是否要求双输出。</param>
    /// <param name="renders">是否要求 PNG。</param>
    private static void ValidateExisting(string recordPath, string prefix, Source source, string sourceHash, bool patches, bool renders)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(recordPath));
        var record = document.RootElement;
        if (record.GetProperty("fingerprint").GetString() != source.Fingerprint || record.GetProperty("source_sha256").GetString() != sourceHash
            || record.GetProperty("source_path").GetString() != Path.GetFullPath(source.Path))
            throw new InvalidDataException($"缓存来源漂移: {source.Path}");
        var files = new List<(string Suffix, string Key)> { (".cv.bin", "cv_sha256") };
        if (patches)
            files.AddRange([(".cls.bin", "cls_sha256"), (".patch.bin", "patches_sha256")]);
        if (renders)
            files.Add((".png", "render_sha256"));
        foreach (var file in files)
        {
            if (record.GetProperty(file.Key).GetString() != HashFile(prefix + file.Suffix))
                throw new InvalidDataException($"缓存输出漂移: {prefix + file.Suffix}");
        }
    }

    /// <summary>按小端 float32 编码有限的 ONNX 数组，返回二进制缓冲。</summary>
    /// <param name="values">模型输出浮点数组。</param>
    /// <returns>小端连续 float32 字节。</returns>
    private static byte[] EncodeFloats(float[] values)
    {
        if (values.Any(value => !float.IsFinite(value)))
            throw new InvalidDataException("ONNX 输出包含非有限值。");
        var bytes = new byte[checked(values.Length * sizeof(float))];
        for (int index = 0; index < values.Length; index++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(index * sizeof(float), sizeof(float)), values[index]);
        return bytes;
    }

    /// <summary>以哈希保护方式原子落盘；已有不同内容必须使用新目录，不静默覆盖。</summary>
    /// <param name="path">输出绝对路径。</param>
    /// <param name="bytes">目标内容。</param>
    private static void WriteImmutable(string path, byte[] bytes)
    {
        if (File.Exists(path))
        {
            if (HashFile(path) != Hash(bytes))
                throw new InvalidDataException($"已有输出与请求不一致: {path}");
            return;
        }
        string temporary = path + ".partial";
        File.WriteAllBytes(temporary, bytes);
        File.Move(temporary, path);
    }

    /// <summary>计算文件 SHA256，返回小写十六进制；源文件不存在直接失败。</summary>
    /// <param name="path">文件路径。</param>
    /// <returns>SHA256 字符串。</returns>
    private static string HashFile(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    /// <summary>计算字节数组 SHA256，返回小写十六进制。</summary>
    /// <param name="bytes">输入内容。</param>
    /// <returns>SHA256 字符串。</returns>
    private static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    /// <summary>固定来源清单条目，字段与 repair_sources.json 一致。</summary>
    private sealed record Source(string Fingerprint, string Path);
}
