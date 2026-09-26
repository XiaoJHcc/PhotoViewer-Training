using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Avalonia.Win32;
using MetadataExtractor;
using MetadataExtractor.Formats.Exif;
using MetadataExtractor.Formats.Exif.Makernotes;
using Microsoft.ML.OnnxRuntime;
using PhotoViewer.Core;
using PhotoViewer.Core.AI;
using PhotoViewer.Core.Database;
using PhotoViewer.Core.Image;

namespace DatasetBuilder;

/// <summary>
/// 训练数据集提取工具（Plan-3-1 M1 数据地基）：清单驱动，把选定文件夹的全部 AI 训练信息
/// 提取到一个**独立、可扩展的数据集库**（非产品 photos.db），产出覆盖率报告 + GATE。
/// 每指纹提取：原片 CLS + 增强 CLS（多视图探针用）+ patch token + CV grid + EXIF/rating + 来源标签。
/// </summary>
internal static class Program
{
    private static int _ingested;
    private static int _skipped;
    private static int _failed;
    private static int _total;
    private static readonly ConcurrentBag<string> _failures = new();

    private static int Main(string[] args)
    {
        IngestManifest manifest;
        int? concurrencyOverride;
        bool scanOnly, noPatch;
        string? modelFile, modelIdOverride, dumpRender, dumpAccel;
        try
        {
            (manifest, concurrencyOverride, scanOnly, modelFile, modelIdOverride, noPatch, dumpRender, dumpAccel) = BuildManifest(args);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] {ex.Message}");
            PrintUsage();
            return 1;
        }
        if (manifest.Folders.Count == 0)
        {
            PrintUsage();
            return 1;
        }

        // 分布探查：只扫描 + 指纹聚合 + 读 EXIF，不解码、不建库、不需要 Avalonia/DirectML。
        if (scanOnly)
        {
            ScanReport.Print(FingerprintGrouper.Scan(manifest.Folders, manifest.ExcludeKeywords));
            return 0;
        }

        // 渲染缓存转储（梯4 LoRA 前置）：用与 DINO 预处理完全相同的解码+缩放路径，
        // 把每组代表件渲成 <fingerprint>.png 训练缓存。不建库、不提特征、无 ONNX 会话。
        if (dumpRender != null)
            return RunDumpRender(manifest, dumpRender, concurrencyOverride);

        // 加速度计/姿态导出（水平度信号基建）：只读 EXIF（机型 + Sony 0x940F），
        // 不解码位图、不建库、无 Avalonia/dispatcher。三轴恒导出，俯仰/横滚仅已校准机型。
        if (dumpAccel != null)
            return RunDumpAccel(manifest, dumpAccel, concurrencyOverride);

        if ((modelFile == null) != (modelIdOverride == null))
        {
            Console.WriteLine("[ERROR] --model-file 与 --model-id 必须成对提供（换模型重提时两行 model_id 必须落对）。");
            return 1;
        }

        if (string.IsNullOrWhiteSpace(manifest.DbPath))
        {
            PrintUsage();
            return 1;
        }

        // Avalonia 平台初始化（RenderTargetBitmap / WriteableBitmap 需要）。软件渲染（Skia CPU）避开 GPU 合成器。
        AppBuilder.Configure<Application>()
            .UsePlatformDetect()
            .With(new Win32PlatformOptions { RenderingMode = new[] { Win32RenderingMode.Software } })
            .SetupWithoutStarting();
        DinoFeatureExtractor.ConfigureSession(o => o.AppendExecutionProvider_DML());

        // 提取在后台线程跑，主线程 pump UI dispatcher：DINO 预处理的 RenderTargetBitmap 会向 UI 线程做
        // 渲染往返，必须有活跃 dispatcher 服务它才不死锁（等同 App 内有运行中的 dispatcher 循环）。
        int rc = 0;
        var cts = new CancellationTokenSource();
        _ = Task.Run(async () =>
        {
            try { rc = await RunPipelineAsync(manifest, concurrencyOverride, modelFile, modelIdOverride, noPatch); }
            catch (Exception ex) { Console.WriteLine($"[ERROR] {ex}"); rc = 3; }
            finally { cts.Cancel(); }
        });
        try { Dispatcher.UIThread.MainLoop(cts.Token); }
        catch (OperationCanceledException) { }
        return rc;
    }

    /// <summary>
    /// --dump-render 入口：初始化 Avalonia（软件渲染）+ pump dispatcher（RenderTargetBitmap 渲染
    /// 会向 UI 线程做往返，同主管线），后台线程跑 <see cref="DumpRenderAsync"/>。无 ONNX 会话，不悬挂。
    /// </summary>
    private static int RunDumpRender(IngestManifest manifest, string dir, int? concurrencyOverride)
    {
        AppBuilder.Configure<Application>()
            .UsePlatformDetect()
            .With(new Win32PlatformOptions { RenderingMode = new[] { Win32RenderingMode.Software } })
            .SetupWithoutStarting();
        int rc = 0;
        var cts = new CancellationTokenSource();
        _ = Task.Run(async () =>
        {
            try { rc = await DumpRenderAsync(manifest, dir, concurrencyOverride); }
            catch (Exception ex) { Console.WriteLine($"[ERROR] {ex}"); rc = 3; }
            finally { cts.Cancel(); }
        });
        try { Dispatcher.UIThread.MainLoop(cts.Token); }
        catch (OperationCanceledException) { }
        return rc;
    }

    /// <summary>
    /// 逐指纹组把代表件按 DINO 输入口径渲染成 PNG 缓存（梯4 LoRA 训练/评估用）：
    /// 与 <see cref="DinoFeatureExtractor.RenderInputBitmap"/> 同一解码+缩放路径，保证训练像素 ≡ 部署像素。
    /// 幂等：已存在非空文件即跳过。
    /// </summary>
    private static async Task<int> DumpRenderAsync(IngestManifest manifest, string dir, int? concurrencyOverride)
    {
        string versionPath = Path.Combine(dir, "decode-version.txt");
        if (System.IO.Directory.Exists(dir) && System.IO.Directory.EnumerateFileSystemEntries(dir).Any())
        {
            if (!File.Exists(versionPath) || File.ReadAllText(versionPath).Trim() != PhotoDecode.Version)
                throw new InvalidDataException("渲染目录包含旧版本或无版本缓存；请使用新的输出目录，保留旧缓存用于对照。");
        }
        System.IO.Directory.CreateDirectory(dir);
        File.WriteAllText(versionPath, PhotoDecode.Version + Environment.NewLine);
        Console.WriteLine("扫描 + 指纹聚合中…");
        var groups = FingerprintGrouper.Scan(manifest.Folders, manifest.ExcludeKeywords);
        Console.WriteLine($"{groups.Count} 指纹组 → 渲染 {DinoModelResources.InputSize}px PNG → {dir}");
        int concurrency = concurrencyOverride ?? manifest.Concurrency ?? Math.Max(1, Environment.ProcessorCount / 2);
        var sem = new SemaphoreSlim(concurrency);
        int done = 0, skip = 0, fail = 0;
        var tasks = groups.Select(g => Task.Run(async () =>
        {
            await sem.WaitAsync();
            try
            {
                string path = Path.Combine(dir, g.Fingerprint + ".png");
                if (File.Exists(path) && new FileInfo(path).Length > 0)
                {
                    Interlocked.Increment(ref skip);
                    return;
                }
                if (!PhotoDecode.CanDecode(g.Representative.Path))
                {
                    Interlocked.Increment(ref fail);
                    Console.WriteLine($"[FAIL] 不可解码: {g.Representative.RelPath}");
                    return;
                }
                using var bitmap = PhotoDecode.LoadBitmap(g.Representative.Path);
                if (bitmap == null)
                {
                    Interlocked.Increment(ref fail);
                    Console.WriteLine($"[FAIL] 解码失败: {g.Representative.RelPath}");
                    return;
                }
                using var rtb = DinoFeatureExtractor.RenderInputBitmap(bitmap);
                using (var fs = File.Create(path))
                    rtb.Save(fs);
                int d = Interlocked.Increment(ref done);
                if (d % 200 == 0 || d == groups.Count)
                    Console.WriteLine($"  [{d}/{groups.Count}] 跳过 {skip} · 失败 {fail}");
            }
            catch (Exception ex)
            {
                Interlocked.Increment(ref fail);
                Console.WriteLine($"[FAIL] {g.Representative.RelPath}: {ex.Message}");
            }
            finally
            {
                sem.Release();
            }
        })).ToArray();
        await Task.WhenAll(tasks);
        Console.WriteLine($"DUMP GATE: done={done} skip={skip} fail={fail} / total={groups.Count}");
        return fail == 0 ? 0 : 2;
    }

    /// <summary>--dump-accel 入口：纯 EXIF 读取（无位图解码/无 Avalonia/无 ONNX），主线程直接跑，不悬挂。</summary>
    private static int RunDumpAccel(IngestManifest manifest, string csvPath, int? concurrencyOverride)
    {
        Console.WriteLine("扫描 + 指纹聚合中…");
        var groups = FingerprintGrouper.Scan(manifest.Folders, manifest.ExcludeKeywords);
        Console.WriteLine($"{groups.Count} 指纹组 → 读取机型 + Sony 0x940F 加速度计 → {csvPath}");

        int concurrency = concurrencyOverride ?? manifest.Concurrency ?? Math.Max(1, Environment.ProcessorCount / 2);
        var sem = new SemaphoreSlim(concurrency);
        var rows = new ConcurrentBag<string>();
        var models = new ConcurrentDictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        int done = 0, withAccel = 0, withAngles = 0, noExif = 0;
        var tasks = groups.Select(g => Task.Run(() =>
        {
            sem.Wait();
            try
            {
                var r = ReadGroupAttitude(g);
                if (r == null)
                {
                    Interlocked.Increment(ref noExif);
                    rows.Add(string.Join(',', Csv(g.Fingerprint), Csv(g.Input.FilenameNoExt),
                        Csv(g.Representative.Folder.EventLabel), Csv(g.Representative.Folder.SubjectLabel),
                        Csv(g.Representative.RelPath), "", 0, "", "", "", "", "", "", "", ""));
                    return;
                }
                var (model, att, hasAccel) = r.Value;
                models.AddOrUpdate(model ?? "", 1, (_, c) => c + 1);
                if (hasAccel) Interlocked.Increment(ref withAccel);
                if (att?.RollDeg != null) Interlocked.Increment(ref withAngles);
                string F(double? v, string fmt) => v.HasValue ? v.Value.ToString(fmt, CultureInfo.InvariantCulture) : "";
                rows.Add(string.Join(',', Csv(g.Fingerprint), Csv(g.Input.FilenameNoExt),
                    Csv(g.Representative.Folder.EventLabel), Csv(g.Representative.Folder.SubjectLabel),
                    Csv(g.Representative.RelPath), Csv(model), hasAccel ? 1 : 0,
                    hasAccel ? att!.Value.RawX.ToString(CultureInfo.InvariantCulture) : "",
                    hasAccel ? att!.Value.RawY.ToString(CultureInfo.InvariantCulture) : "",
                    hasAccel ? att!.Value.RawZ.ToString(CultureInfo.InvariantCulture) : "",
                    F(hasAccel ? att!.Value.Ax : null, "F4"), F(hasAccel ? att!.Value.Ay : null, "F4"),
                    F(hasAccel ? att!.Value.Az : null, "F4"),
                    F(att?.PitchDeg, "F3"), F(att?.RollDeg, "F3")));
                int d = Interlocked.Increment(ref done);
                if (d % 500 == 0 || d == groups.Count)
                    Console.WriteLine($"  [{d}/{groups.Count}] 有加速度 {withAccel} · 有姿态角 {withAngles}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[FAIL] {g.Representative.RelPath}: {ex.Message}");
            }
            finally
            {
                sem.Release();
            }
        })).ToArray();
        Task.WaitAll(tasks);

        System.IO.Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(csvPath))!);
        using (var w = new StreamWriter(csvPath, false, new System.Text.UTF8Encoding(false)))
        {
            w.WriteLine("fingerprint,filename_noext,event_label,subject_label,source_rel_path,camera_model," +
                        "has_accel,raw_x,raw_y,raw_z,accel_x_ms2,accel_y_ms2,accel_z_ms2,pitch_deg,roll_deg");
            foreach (var row in rows) w.WriteLine(row);
        }

        Console.WriteLine($"DUMP-ACCEL GATE: groups={groups.Count} withAccel={withAccel} withAngles={withAngles} noExif={noExif}");
        foreach (var kv in models.OrderByDescending(kv => kv.Value))
            Console.WriteLine($"  机型 {kv.Key}: {kv.Value} 组");
        return 0;
    }

    /// <summary>CSV 字段转义（含逗号/引号/换行才加引号）。null → 空串。</summary>
    private static string Csv(string? s)
    {
        if (string.IsNullOrEmpty(s)) return "";
        return s.Contains(',') || s.Contains('"') || s.Contains('\n')
            ? "\"" + s.Replace("\"", "\"\"") + "\""
            : s;
    }

    /// <summary>
    /// 读一组的姿态信息：按解码代价升序逐文件读 EXIF，直到拿到 0x940F 或试完。
    /// 机型取首个有 IFD0 Model 的文件。返回 null = 全组连 EXIF 都读不到。
    /// </summary>
    private static (string? Model, AccelAttitude? Att, bool HasAccel)? ReadGroupAttitude(FpGroup group)
    {
        string? model = null;
        AccelAttitude? att = null;
        bool hasAccel = false, anyExif = false;
        foreach (var file in group.Files)
        {
            IReadOnlyList<MetadataExtractor.Directory> dirs;
            try
            {
                using var stream = File.OpenRead(file.Path);
                dirs = ImageMetadataReader.ReadMetadata(stream);
            }
            catch { continue; }
            anyExif = true;
            model ??= dirs.OfType<ExifIfd0Directory>().FirstOrDefault()
                ?.GetString(ExifDirectoryBase.TagModel)?.Trim();
            var sony = dirs.OfType<SonyType1MakernoteDirectory>().FirstOrDefault();
            if (sony != null && SonyAttitudeDecoder.TryDecode(sony, model, out var a))
            {
                att = a;
                hasAccel = true;
                break; // 拿到加速度即停（同组同曝光，任意文件等价）
            }
        }
        return anyExif ? (model, att, hasAccel) : null;
    }

    /// <summary>核心流水线（在后台线程执行）：建库 → 写 meta → 扫描聚合 → 逐组四路提取 → 覆盖率报告 + GATE。</summary>
    private static async Task<int> RunPipelineAsync(
        IngestManifest manifest, int? concurrencyOverride, string? modelFile, string? modelIdOverride, bool noPatch)
    {
        // 升级梯实验：外部 ONNX 文件替代内置资源模型（须与 --model-id 成对，见 Main 校验）。
        if (modelFile != null)
        {
            Console.WriteLine($"模型覆盖: {modelFile} → model_id={modelIdOverride} · noPatch={noPatch}");
            DinoFeatureExtractor.ConfigureModelOverride(File.ReadAllBytes(modelFile));
        }
        bool enhance = manifest.Enhance?.Enabled ?? true;
        string modelId = $"{modelIdOverride ?? DinoModelResources.ModelId}+decode-{PhotoDecode.Version}";
        // 后缀编入 ClipFactor 与色彩模型标记 ycc（YCbCr 保色度重建）+ SaturationScale：算法语义变了后缀即变 → 缓存自动失效、永不漂移。
        string? enhancedModelId = enhance
            ? $"{modelId}+clhe{ImageEnhancer.ClipFactor.ToString("0.0", CultureInfo.InvariantCulture)}ycc{ImageEnhancer.SaturationScale.ToString("0.0", CultureInfo.InvariantCulture)}"
            : null;
        string cvSpec = $"{CvGridResult.CurrentVersion}+decode-{PhotoDecode.Version}";

        var dbPath = Path.GetFullPath(manifest.DbPath);
        var db = new DatasetDatabase(dbPath);
        db.Initialize();
        await db.WriteMetaAsync(new Dictionary<string, string>
        {
            ["dino_model_id"] = modelId,
            ["enhanced_model_id"] = enhancedModelId ?? "(disabled)",
            ["clip_factor"] = ImageEnhancer.ClipFactor.ToString("0.0", CultureInfo.InvariantCulture),
            ["saturation_scale"] = ImageEnhancer.SaturationScale.ToString("0.0", CultureInfo.InvariantCulture),
            ["color_model"] = "ycc-constant-chroma (ch' = Y' + s*(ch - Y))",
            ["cv_spec"] = cvSpec,
            ["dino_input_size"] = DinoModelResources.InputSize.ToString(CultureInfo.InvariantCulture),
            ["decode_version"] = PhotoDecode.Version,
            // 一致性冻结点（Plan-3-1 §1.2 清单②）：增强施加在与 DINO/CV 同一张全分辨率解码位图上。
            ["enhance_resolution"] = "full-res-decode (same bitmap fed to DINO squash-to-518 and CV)",
        });
        Console.WriteLine($"数据集库: {dbPath}");
        Console.WriteLine($"原片 model_id={modelId} · 增强={(enhancedModelId ?? "(disabled)")} · cv={cvSpec}");

        // 精修清单（可选）：命中的直出原片文件名去扩展名标 is_retouched=1；提供了清单则非命中标 0，否则全 NULL。
        var (retouched, hasRetouchedList) = LoadRetouchedList(manifest.RetouchedList);

        Console.WriteLine("扫描 + 指纹聚合中…");
        var groups = FingerprintGrouper.Scan(manifest.Folders, manifest.ExcludeKeywords);
        _total = groups.Count;
        Console.WriteLine($"扫描到 {groups.Sum(g => g.Files.Count)} 文件 → {_total} 指纹组");
        if (_total == 0) return 0;

        var sw = Stopwatch.StartNew();
        int concurrency = concurrencyOverride ?? manifest.Concurrency ?? Math.Max(1, Environment.ProcessorCount / 2);
        var sem = new SemaphoreSlim(concurrency);
        var tasks = groups.Select(g => Task.Run(() => ProcessGroupAsync(
            g, db, modelId, enhancedModelId, cvSpec, retouched, hasRetouchedList, noPatch, sem))).ToArray();
        await Task.WhenAll(tasks);
        sw.Stop();

        bool gate = await CoverageReport.GenerateAsync(
            db, groups, modelId, enhancedModelId, cvSpec,
            _ingested, _skipped, _failed, _failures.ToArray(), sw.Elapsed, noPatch: noPatch);

        return gate ? 0 : 2;
    }

    /// <summary>处理一个指纹组：评估缺失 → 解码代表 → 四路按需提取 → 单事务写库。</summary>
    private static async Task ProcessGroupAsync(
        FpGroup group, DatasetDatabase db, string modelId, string? enhancedModelId, string cvSpec,
        HashSet<string> retouched, bool hasRetouchedList, bool noPatch, SemaphoreSlim sem)
    {
        await sem.WaitAsync();
        try
        {
            var rep = group.Representative;
            var exif = BuildExifSnapshot(rep.Exif, group.Rating);
            // 精修命中按代表文件名去扩展名判定；有清单则命中 1 / 未命中 0，无清单则 NULL（未知）。
            int? isRetouched = hasRetouchedList
                ? (retouched.Contains(Path.GetFileNameWithoutExtension(rep.Path)) ? 1 : 0)
                : null;
            var source = new PhotoSourceInfo(rep.RelPath, rep.Folder.EventLabel, rep.Folder.SubjectLabel, group.Formats, isRetouched);

            // 不可解码（仅 RAW 组）：只写身份 + EXIF + 来源标签。
            if (!PhotoDecode.CanDecode(rep.Path))
            {
                await db.WriteIndexedAsync(group.Input, group.Fingerprint, modelId, enhancedModelId,
                    null, null, null, null, null, 0, 0, exif, source);
                Interlocked.Increment(ref _ingested);
                ReportProgress();
                return;
            }

            var missing = await db.EvaluateMissingAsync(group.Fingerprint, modelId, enhancedModelId, cvSpec, includePatch: !noPatch);
            if (!missing.AnyMissing)
            {
                // 全齐备：仍刷新 EXIF/rating/来源标签（清单可能更新了标签），但不重算。
                await db.WriteIndexedAsync(group.Input, group.Fingerprint, modelId, enhancedModelId,
                    null, null, null, null, null, 0, 0, exif, source);
                Interlocked.Increment(ref _skipped);
                ReportProgress();
                return;
            }

            Bitmap? bitmap = null;
            Bitmap? enhanced = null;
            try
            {
                bitmap = PhotoDecode.LoadBitmap(rep.Path);
                if (bitmap == null)
                {
                    RecordFailure($"解码失败: {rep.RelPath}");
                    ReportProgress();
                    return;
                }

                byte[]? origCls = null, enhCls = null, patch = null, cv = null;
                string? cvSpecWritten = null;
                int cvW = 0, cvH = 0;

                if (missing.NeedOrigCls || missing.NeedPatch)
                {
                    var (cls, patches) = await DinoFeatureExtractor.ExtractDualAsync(bitmap, includePatches: missing.NeedPatch);
                    if (missing.NeedOrigCls) origCls = EncodeFloats(cls);
                    if (missing.NeedPatch && patches != null) patch = EncodeFloats(patches);
                }

                if (missing.NeedEnhCls && enhancedModelId != null)
                {
                    enhanced = ImageEnhancer.Enhance(bitmap);
                    var enhVec = await DinoFeatureExtractor.ExtractAsync(enhanced);
                    enhCls = EncodeFloats(enhVec);
                }

                if (missing.NeedCv)
                {
                    var cvResult = await CvGridExtractor.ExtractAsync(bitmap);
                    cv = cvResult.Encode();
                    cvSpecWritten = cvSpec;
                    cvW = bitmap.PixelSize.Width;
                    cvH = bitmap.PixelSize.Height;
                }

                await db.WriteIndexedAsync(group.Input, group.Fingerprint, modelId, enhancedModelId,
                    origCls, enhCls, patch, cv, cvSpecWritten, cvW, cvH, exif, source);
                Interlocked.Increment(ref _ingested);
                ReportProgress();
            }
            finally
            {
                enhanced?.Dispose();
                bitmap?.Dispose();
            }
        }
        catch (Exception ex)
        {
            RecordFailure($"{group.Representative.RelPath}: {ex.Message}");
            ReportProgress();
        }
        finally
        {
            sem.Release();
        }
    }

    private static ExifSnapshot BuildExifSnapshot(PhotoExif exif, int rating)
    {
        double? cropFactor = null;
        if (exif.FocalLength is > 0 && exif.EquivFocalLength is > 0)
            cropFactor = exif.EquivFocalLength.Value / exif.FocalLength.Value;
        return new ExifSnapshot
        {
            FocalLength = exif.FocalLength,
            Aperture = exif.Aperture,
            ShutterSpeed = exif.ShutterSpeed,
            CropFactor = cropFactor,
            Rating = rating > 0 ? rating : null,
        };
    }

    private static void RecordFailure(string msg)
    {
        Console.WriteLine($"[FAIL] {msg}");
        _failures.Add(msg);
        Interlocked.Increment(ref _failed);
    }

    private static void ReportProgress()
    {
        int done = _ingested + _skipped + _failed;
        if (done % 20 == 0 || done == _total)
            Console.WriteLine($"  [{done}/{_total}] 入库 {_ingested} · 跳过 {_skipped} · 失败 {_failed}");
    }

    private static byte[] EncodeFloats(float[] data)
    {
        var bytes = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static (HashSet<string> set, bool hasList) LoadRetouchedList(string? path)
    {
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (string.IsNullOrWhiteSpace(path)) return (set, false);
        if (!File.Exists(path))
        {
            Console.WriteLine($"[WARN] 精修清单不存在，忽略: {path}");
            return (set, false);
        }
        foreach (var line in File.ReadLines(path))
        {
            var name = line.Trim();
            if (name.Length == 0) continue;
            set.Add(Path.GetFileNameWithoutExtension(name));
        }
        return (set, true);
    }

    private static (IngestManifest, int?, bool, string?, string?, bool, string?, string?) BuildManifest(string[] args)
    {
        string? manifestPath = null, dbPath = null;
        int? concurrency = null;
        bool noEnhance = false, scanOnly = false;
        string? modelFile = null, modelIdOverride = null; bool noPatch = false;
        string? dumpRender = null, dumpAccel = null;
        var folders = new List<string>();

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--manifest" when i + 1 < args.Length: manifestPath = args[++i]; break;
                case "--db" when i + 1 < args.Length: dbPath = args[++i]; break;
                case "--concurrency" when i + 1 < args.Length: concurrency = int.Parse(args[++i]); break;
                case "--no-enhance": noEnhance = true; break;
                case "--scan-only": scanOnly = true; break;
                case "--model-file" when i + 1 < args.Length: modelFile = args[++i]; break;
                case "--model-id" when i + 1 < args.Length: modelIdOverride = args[++i]; break;
                case "--no-patch": noPatch = true; break;
                case "--dump-render" when i + 1 < args.Length: dumpRender = args[++i]; break;
                case "--dump-accel" when i + 1 < args.Length: dumpAccel = args[++i]; break;
                default:
                    if (!args[i].StartsWith("--")) folders.Add(args[i]);
                    break;
            }
        }

        if (manifestPath != null)
        {
            var loaded = IngestManifest.Load(manifestPath);
            // 清单之外的裸文件夹追加为无标签条目（--dump-accel 补扫旧批等场景）。
            foreach (var f in folders)
                loaded.Folders.Add(new FolderEntry { Path = f, Recursive = true });
            return (loaded, concurrency, scanOnly, modelFile, modelIdOverride, noPatch, dumpRender, dumpAccel);
        }

        // 快速模式：直接给文件夹 + --db，无标签。--scan-only / --dump-render / --dump-accel 不写库，故不要求 --db。
        if (folders.Count == 0)
            throw new ArgumentException("需要 --manifest <路径>，或提供文件夹（+ --db <路径>，--scan-only 除外）。");
        if (dbPath == null && !scanOnly && dumpRender == null && dumpAccel == null)
            throw new ArgumentException("快速模式需要 --db <数据集库路径>（或加 --scan-only 只看分布）。");

        var m = new IngestManifest
        {
            DbPath = dbPath ?? "",
            Enhance = new EnhanceOptions { Enabled = !noEnhance },
            Folders = folders.Select(f => new FolderEntry { Path = f, Recursive = true }).ToList(),
        };
        return (m, concurrency, scanOnly, modelFile, modelIdOverride, noPatch, dumpRender, dumpAccel);
    }

    private static void PrintUsage()
    {
        Console.WriteLine("DatasetBuilder — 训练数据集提取工具 (Plan-3-1 M1)");
        Console.WriteLine();
        Console.WriteLine("用法:");
        Console.WriteLine("  dotnet run --project Training/DatasetBuilder -- --manifest <manifest.json>");
        Console.WriteLine("  dotnet run --project Training/DatasetBuilder -- <folder>... --db <dataset.db> [--no-enhance]");
        Console.WriteLine("  dotnet run --project Training/DatasetBuilder -- <folder>... --scan-only");
        Console.WriteLine();
        Console.WriteLine("  --manifest <path>   清单驱动（推荐）：dbPath + folders(带事件/题材标签) + 可选 retouchedList");
        Console.WriteLine("  --db <path>         快速模式的数据集库路径");
        Console.WriteLine("  --scan-only         只扫描 + 指纹聚合 + 读 EXIF，输出焦段/星级/时间分布（不建库、不提特征）");
        Console.WriteLine("  --no-enhance        不提取增强 CLS（仅原片路）");
        Console.WriteLine("  --concurrency <n>   解码并发（默认 CPU/2）");
        Console.WriteLine("  --model-file <path> 升级梯实验：改用指定 ONNX 文件替代内置模型（须与 --model-id 成对）");
        Console.WriteLine("  --model-id <id>     升级梯实验：本批特征落库的 model_id（须与 --model-file 成对）");
        Console.WriteLine("  --no-patch          不提取 patch token（梯级探针只提双路 CLS，省 ~38GB）");
        Console.WriteLine("  --dump-render <dir> 梯4 LoRA 前置：按 DINO 输入口径把每组代表件渲成 <fingerprint>.png 缓存（不建库、不提特征）");
        Console.WriteLine("  --dump-accel <csv>  姿态导出：逐指纹组读机型 + Sony 0x940F 加速度计 → CSV（三轴恒有，俯仰/横滚仅已校准机型；不建库、不解码位图）");
    }
}
