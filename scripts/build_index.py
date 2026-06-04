"""
批量索引脚本 — 基于 FileProcessor 模块化入库
支持三种模式:
  1. 单文件:     python scripts/build_index.py --file "C:/path/doc.pdf"
  2. 多文件:     python scripts/build_index.py --files "f1.pdf" "f2.pdf"
  3. 目录批量:   python scripts/build_index.py --dir "D:/知识库/变电/标准规范"
                 (递归扫描目录, 逐个调用 FileProcessor)

用法:
  python scripts/build_index.py --file "D:/test.pdf"
  python scripts/build_index.py --files "a.pdf" "b.pdf" --domain 变电
  python scripts/build_index.py --dir "D:/知识库/变电" --limit 100
  python scripts/build_index.py --list                           # 列出入库文件
  python scripts/build_index.py --summary                        # 查看统计
  python scripts/build_index.py --delete "hash或文件名"
  python scripts/build_index.py --reindex "hash或文件名"
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ingestion.file_processor import FileProcessor, FileStatus


def cmd_add_file(args):
    """添加单个文件"""
    processor = FileProcessor()
    print(f"📄 处理单个文件: {args.file}")
    result = processor.process(args.file, domain=args.domain, category=args.category,
                               progress_callback=_progress)
    _print_result(result)


def cmd_add_files(args):
    """添加多个文件"""
    processor = FileProcessor()
    print(f"📦 处理 {len(args.files)} 个文件...")
    batch = processor.process_batch(args.files, domain=args.domain, category=args.category,
                                    progress_callback=_progress)
    _print_batch(batch)


def cmd_add_dir(args):
    """从目录批量添加"""
    import glob as globmod
    processor = FileProcessor()

    # 收集支持的文件
    exts = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ofd", ".txt", ".md"}
    files = []
    for root, dirs, filenames in os.walk(args.dir):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(os.path.join(root, fn))

    print(f"📂 目录: {args.dir}")
    print(f"   发现 {len(files)} 个可处理文件")

    if args.limit and args.limit > 0:
        files = files[:args.limit]
        print(f"   限制处理: {len(files)} 个")

    batch = processor.process_batch(files, domain=args.domain, category=args.category,
                                    progress_callback=_progress)
    _print_batch(batch)


def cmd_list(args):
    """列出已入库文件"""
    processor = FileProcessor()
    st = args.status or None
    dm = args.domain or None
    files = processor.list_files(status=st, domain=dm, limit=args.limit or 50)

    print(f"\n{'='*70}")
    print(f"{'#':<4} {'文件名':<40} {'状态':<12} {'Chunks':<8} {'域'}")
    print(f"{'='*70}")
    for i, f in enumerate(files):
        print(f"{i+1:<4} {f['file_name'][:39]:<40} {f['status']:<12} "
              f"{f['chunks_count']:<8} {f.get('domain','-') or '-'}")
    print(f"{'='*70}")
    print(f"显示 {len(files)} 条")


def cmd_summary(args):
    """显示入库统计"""
    processor = FileProcessor()
    s = processor.get_summary()
    print(f"\n{'='*40}")
    print(f"📊 索引入库统计")
    print(f"{'='*40}")
    print(f"总文件数: {s['total_files']}")
    print(f"  已完成: {s['by_status'].get('completed', 0)}")
    print(f"  失败:   {s['by_status'].get('failed', 0)}")
    print(f"  已删除: {s['by_status'].get('deleted', 0)}")
    print(f"总 Chunks: {s['total_chunks']:,}")
    print(f"总字符数: {s['total_chars']:,}")
    print(f"\n按域分布:")
    for dm, cnt in sorted(s.get('by_domain', {}).items(), key=lambda x: -x[1]):
        print(f"  {dm}: {cnt} 个文件")


def cmd_delete(args):
    """删除文件"""
    processor = FileProcessor()
    ok = processor.delete(args.identifier)
    print(f"✅ 已删除: {args.identifier}" if ok else f"❌ 未找到: {args.identifier}")


def cmd_reindex(args):
    """重建索引"""
    processor = FileProcessor()
    result = processor.reindex(args.identifier, progress_callback=_progress)
    _print_result(result)


def _progress(stage: str, pct: float):
    bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
    print(f"  [{bar}] {stage} ({pct*100:.0f}%)")


def _print_result(r):
    if r.status == FileStatus.COMPLETED:
        print(f"\n✅ 成功: {r.file_name}")
        print(f"   Hash: {r.file_hash[:16]}...")
        print(f"   域: {r.domain} | 类目: {r.category} | 编号: {r.doc_number}")
        print(f"   Chunks: {r.chunks_created} | 字符: {r.chars_extracted}")
        print(f"   解析: {r.parse_time_ms:.0f}ms | 嵌入: {r.embed_time_ms:.0f}ms | 总: {r.total_time_ms:.0f}ms")
    else:
        print(f"\n❌ 失败: {r.file_name}")
        print(f"   错误: {r.error_message}")


def _print_batch(b):
    print(f"\n{'='*50}")
    print(f"📦 批量处理完成")
    print(f"   总数: {b.total} | 成功: {b.success} | 失败: {b.failed}")
    print(f"   总耗时: {b.total_time_ms:.0f}ms")
    for r in b.results:
        icon = "✅" if r.status == FileStatus.COMPLETED else "❌"
        print(f"   {icon} {r.file_name[:45]} | {r.chunks_created} chunks | {r.domain or '-'}")


def main():
    parser = argparse.ArgumentParser(description="模块化知识库索引工具")
    sub = parser.add_subparsers(dest="command", help="操作命令")

    # add
    p_file = sub.add_parser("add-file", help="添加单个文件")
    p_file.add_argument("--file", required=True, help="文件路径")
    p_files = sub.add_parser("add-files", help="添加多个文件")
    p_files.add_argument("--files", nargs="+", required=True, help="文件路径列表")
    p_dir = sub.add_parser("add-dir", help="从目录批量添加")
    p_dir.add_argument("--dir", required=True, help="目录路径")
    for p in [p_file, p_files, p_dir]:
        p.add_argument("--domain", default=None, help="指定专业域")
        p.add_argument("--category", default=None, help="指定类目")
    p_dir.add_argument("--limit", type=int, default=None, help="限制文件数")

    # list
    p_list = sub.add_parser("list", help="列出已入库文件")
    p_list.add_argument("--status", default=None, help="按状态过滤")
    p_list.add_argument("--domain", default=None, help="按域过滤")
    p_list.add_argument("--limit", type=int, default=50, help="条数限制")

    # summary
    sub.add_parser("summary", help="入库统计")

    # delete
    p_del = sub.add_parser("delete", help="删除文件")
    p_del.add_argument("identifier", help="文件 hash 或路径")

    # reindex
    p_reidx = sub.add_parser("reindex", help="重建文件索引")
    p_reidx.add_argument("identifier", help="文件 hash 或路径")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    dispatch = {
        "add-file": cmd_add_file,
        "add-files": cmd_add_files,
        "add-dir": cmd_add_dir,
        "list": cmd_list,
        "summary": cmd_summary,
        "delete": cmd_delete,
        "reindex": cmd_reindex,
    }

    t0 = time.time()
    dispatch[args.command](args)
    print(f"\n⏱ 总耗时: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
