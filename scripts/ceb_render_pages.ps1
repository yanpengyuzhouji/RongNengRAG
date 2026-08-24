param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [string]$ApabiDir = "D:\Apabi reader",
    [int]$Width = 800,
    [int]$Height = 1000
)

$ErrorActionPreference = "Stop"

# Apabi Reader 4.x ships a 32-bit ActiveX control.  The caller deliberately
# starts this script through SysWOW64 Windows PowerShell so the COM class is
# visible.  The control is isolated in this short-lived process because it is
# not safe to load the 32-bit COM runtime into the Python worker.
$source = @"
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;

public sealed class CebViewerHost : AxHost
{
    public CebViewerHost() : base("{2CF3F79E-4B63-48C6-B368-EE611B7024ED}") { }
    public object Viewer { get { return GetOcx(); } }
}

public static class CebPageRenderer
{
    [DllImport("user32.dll")]
    private static extern bool PrintWindow(IntPtr hwnd, IntPtr hdcBlt, uint flags);

    public static int Render(string inputPath, string outputDir, string apabiDir, int width, int height)
    {
        Directory.CreateDirectory(outputDir);
        Environment.CurrentDirectory = apabiDir;
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);

        using (var form = new Form())
        using (var host = new CebViewerHost())
        {
            form.ShowInTaskbar = false;
            form.FormBorderStyle = FormBorderStyle.None;
            form.StartPosition = FormStartPosition.Manual;
            form.Location = new Point(0, 0);
            form.ClientSize = new Size(width, height);

            host.Dock = DockStyle.Fill;
            host.BackColor = Color.White;
            form.Controls.Add(host);
            form.Show();
            host.CreateControl();

            dynamic viewer = host.Viewer;
            if (!viewer.CEBViewerInit())
                throw new InvalidOperationException("CEBViewerInit failed");

            try
            {
                int openResult = viewer.OpenCEBFile(inputPath, true);
                int totalPages = viewer.GetTotalPageNum();
                if (openResult != 0 || totalPages <= 0)
                    throw new InvalidOperationException(
                        "OpenCEBFile failed: result=" + openResult + ", pages=" + totalPages);

                for (int page = 1; page <= totalPages; page++)
                {
                    viewer.GotoPage(page, true);
                    Application.DoEvents();
                    host.Refresh();
                    Application.DoEvents();
                    Thread.Sleep(120);

                    string outputPath = Path.Combine(
                        outputDir, page.ToString("D4") + ".png");
                    using (var bitmap = new Bitmap(
                        width, height, PixelFormat.Format24bppRgb))
                    using (var graphics = Graphics.FromImage(bitmap))
                    {
                        // PrintWindow may leave pixels outside the visible
                        // ActiveX viewport untouched.  White is the CEB page
                        // background and avoids black borders entering OCR.
                        graphics.Clear(Color.White);
                        IntPtr hdc = graphics.GetHdc();
                        try
                        {
                            if (!PrintWindow(host.Handle, hdc, 2))
                                throw new InvalidOperationException(
                                    "PrintWindow failed for page " + page);
                        }
                        finally
                        {
                            graphics.ReleaseHdc(hdc);
                        }
                        bitmap.Save(outputPath, ImageFormat.Png);
                    }
                }
                return totalPages;
            }
            finally
            {
                try { viewer.CloseCEBFile(); } catch { }
                try { viewer.CEBViewerUnInit(); } catch { }
                form.Close();
            }
        }
    }
}
"@

Add-Type -TypeDefinition $source -ReferencedAssemblies @(
    "System.Windows.Forms.dll",
    "System.Drawing.dll",
    "Microsoft.CSharp.dll"
)

$count = [CebPageRenderer]::Render($InputPath, $OutputDir, $ApabiDir, $Width, $Height)
Write-Output ("rendered_pages=" + $count)
