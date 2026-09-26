using System;
using System.IO;
using System.Linq;
using Avalonia;
using Avalonia.Media.Imaging;
using LibHeifSharp;
using MetadataExtractor;
using MetadataExtractor.Formats.Exif;
using MetadataExtractor.Formats.Xmp;
using PhotoViewer.Core.Image;

namespace DatasetBuilder;

/// <summary>
/// CLI 侧的裸文件 EXIF 读取 + 位图解码（不走产品的 ImageFile / IStorageFile 机制）。
/// 逻辑照搬自旧 BatchIngest：MetadataExtractor 直读 EXIF/XMP，LibHeifSharp 解 HEIF。
/// </summary>
public static class PhotoDecode
{
    /// <summary>显示朝向解码版本；HEIF 平面尺寸和 JPEG EXIF 方向修复必须隔离旧缓存。</summary>
    public const string Version = "display-plane-v3";

    /// <summary>可解码为位图的扩展名（RAW 只入身份/EXIF，不解码）。</summary>
    public static readonly string[] DecodableExtensions =
        [".jpg", ".jpeg", ".heif", ".heic", ".hif"];

    /// <summary>入库扫描接受的扩展名。</summary>
    public static readonly string[] ImageExtensions =
        [".jpg", ".jpeg", ".heif", ".heic", ".hif", ".arw"];

    /// <summary>是否为受支持图片扩展名。</summary>
    public static bool IsImage(string path) =>
        ImageExtensions.Contains(Path.GetExtension(path).ToLowerInvariant());

    /// <summary>是否可解码为位图。</summary>
    public static bool CanDecode(string path) =>
        DecodableExtensions.Contains(Path.GetExtension(path).ToLowerInvariant());

    /// <summary>读取一个文件的 EXIF/XMP：拍摄时间、SubSec、焦距/光圈/快门/等效焦距、星级。失败返回全空。</summary>
    /// <param name="path">文件绝对路径。</param>
    /// <returns>EXIF 快照。</returns>
    public static PhotoExif ReadExif(string path)
    {
        try
        {
            using var stream = File.OpenRead(path);
            var dirs = ImageMetadataReader.ReadMetadata(stream);
            var sub = FindShootingExifSubIfd(dirs);
            var xmp = dirs.OfType<XmpDirectory>().FirstOrDefault();

            DateTime? captureTime = null;
            if (sub != null && sub.TryGetDateTime(ExifDirectoryBase.TagDateTimeOriginal, out var dt))
                captureTime = dt;

            const int tagSubSec = 37521; // SubSecTimeOriginal
            string? subSec = sub?.GetString(tagSubSec);

            double? focal = GetRational(sub, ExifDirectoryBase.TagFocalLength);
            double? aperture = GetRational(sub, ExifDirectoryBase.TagFNumber);
            double? shutter = GetRational(sub, ExifDirectoryBase.TagExposureTime);
            // 0xA405 等效焦距按 EXIF 规范是 SHORT——TryGetRational 吃不到（crop_factor 曾全库 0% 的根因）；
            // 对齐产品 ExifLoader/PhotoStatsService：先 int32 后 rational。
            double? equivFocal = GetInt32OrRational(sub, ExifDirectoryBase.Tag35MMFilmEquivFocalLength);

            int rating = 0;
            if (xmp?.XmpMeta != null)
            {
                try
                {
                    var s = xmp.XmpMeta.GetPropertyString("http://ns.adobe.com/xap/1.0/", "xmp:Rating");
                    if (s != null) int.TryParse(s, out rating);
                }
                catch { }
            }

            return new PhotoExif(captureTime, subSec, focal, aperture, shutter, equivFocal, rating);
        }
        catch
        {
            return default;
        }
    }

    /// <summary>解码为显示朝向全分辨率位图：HEIF 使用已旋转平面，JPEG 应用 EXIF 旋转/镜像。</summary>
    /// <param name="path">文件绝对路径。</param>
    /// <returns>已解码位图，调用方负责 Dispose。</returns>
    public static Bitmap? LoadBitmap(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        if (ext is ".heif" or ".heic" or ".hif")
            return DecodeHeif(path);
        using var stream = File.OpenRead(path);
        var orientation = ImageOrientationInfo.FromDirectories(ImageMetadataReader.ReadMetadata(stream), false);
        stream.Position = 0;
        var bitmap = new Bitmap(stream);
        if (orientation.RotationDegreesCw == 0 && !orientation.MirrorHorizontal)
            return bitmap;
        try
        {
            return ThumbnailService.ApplyOrientation(bitmap, orientation.RotationDegreesCw, orientation.MirrorHorizontal)
                ?? throw new InvalidDataException($"无法应用图像方向: {path}");
        }
        finally
        {
            bitmap.Dispose();
        }
    }

    /// <summary>
    /// 在多个 ExifSubIfdDirectory 中选出含拍摄参数的那一个（逻辑对齐产品 ExifLoader.FindShootingExifSubIfd）。
    /// Sony ARW 等 RAW 常有两个 SubIFD：第一个描述 RAW 结构（无 DateTimeOriginal），第二个才含拍摄参数 + 时间。
    /// 若用 FirstOrDefault 会抓错目录 → 拿不到拍摄时间 → 指纹与同曝光 HEIF/JPG 对不上。
    /// </summary>
    private static ExifSubIfdDirectory? FindShootingExifSubIfd(IReadOnlyList<MetadataExtractor.Directory> dirs)
    {
        var subIfds = dirs.OfType<ExifSubIfdDirectory>().ToList();
        if (subIfds.Count == 0) return null;
        if (subIfds.Count == 1) return subIfds[0];
        foreach (var s in subIfds)
        {
            if (s.ContainsTag(ExifDirectoryBase.TagExposureTime) ||
                s.ContainsTag(ExifDirectoryBase.TagFNumber) ||
                s.ContainsTag(ExifDirectoryBase.TagIsoEquivalent))
                return s;
        }
        return subIfds.OrderByDescending(d => d.Tags.Count).First();
    }

    private static double? GetRational(MetadataExtractor.Directory? dir, int tag)
    {
        if (dir == null) return null;
        if (!dir.TryGetRational(tag, out var r)) return null;
        if (r.Denominator == 0) return null;
        return r.ToDouble();
    }

    /// <summary>先按 int32 读、再退 rational——用于 EXIF 规范里 SHORT 型的标签（如 0xA405 等效焦距）。</summary>
    private static double? GetInt32OrRational(MetadataExtractor.Directory? dir, int tag)
    {
        if (dir == null) return null;
        if (dir.TryGetInt32(tag, out var i)) return i;
        if (dir.TryGetRational(tag, out var r) && r.Denominator != 0) return r.ToDouble();
        return null;
    }

    /// <summary>按解码平面的实际尺寸和行步长复制 HEIF，避免旋转后的宽高与容器尺寸不一致。</summary>
    /// <param name="path">HEIF 文件绝对路径。</param>
    /// <returns>显示朝向位图，由调用方释放；解码失败返回 null。</returns>
    private static unsafe Bitmap? DecodeHeif(string path)
    {
        var data = File.ReadAllBytes(path);
        using var context = new HeifContext(data);
        using var handle = context.GetPrimaryImageHandle();
        if (handle == null) return null;
        using var image = handle.Decode(HeifColorspace.Rgb, HeifChroma.InterleavedRgb24);
        if (image == null) return null;

        var plane = image.GetPlane(HeifChannel.Interleaved);
        int w = plane.Width, h = plane.Height;
        int stride = (int)plane.Stride;
        if (w <= 0 || h <= 0 || stride < checked(w * 3) || plane.Scan0 == IntPtr.Zero)
            throw new InvalidDataException($"无效 HEIF 解码平面: {path}");

        var wb = new WriteableBitmap(new PixelSize(w, h), new Vector(96, 96), Avalonia.Platform.PixelFormats.Bgra8888);
        using var locked = wb.Lock();
        var src = (byte*)plane.Scan0;
        var dst = (byte*)locked.Address;
        int dstStride = locked.RowBytes;
        for (int y = 0; y < h; y++)
        {
            var srcRow = src + y * stride;
            var dstRow = dst + y * dstStride;
            for (int x = 0; x < w; x++)
            {
                int si = x * 3, di = x * 4;
                dstRow[di + 0] = srcRow[si + 2]; // B
                dstRow[di + 1] = srcRow[si + 1]; // G
                dstRow[di + 2] = srcRow[si + 0]; // R
                dstRow[di + 3] = 255;            // A
            }
        }
        return wb;
    }
}

/// <summary>一个文件的 EXIF 快照（裸读）。</summary>
public readonly record struct PhotoExif(
    DateTime? CaptureTime, string? SubSec,
    double? FocalLength, double? Aperture, double? ShutterSpeed, double? EquivFocalLength,
    int Rating);
