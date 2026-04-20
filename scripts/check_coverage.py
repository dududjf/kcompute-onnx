"""
代码覆盖率检查工具：
  python scripts/check_coverage.py                     # 检查所有算子
  python scripts/check_coverage.py kop_acos kop_gru    # 检查指定(一个 / 多个)算子
  可选参数: --no-html: 不生成HTML报告(默认生成)  --no-branch: 禁用分支覆盖率(默认启用)
  输出目录: coverage_report/html/index.html (报告) | coverage_report/static/ (页面静态资源)
"""
import argparse, sys, shutil, runpy, coverage
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC, TEST = ROOT / "src" / "kp_onnx_ssbo", ROOT / "src" / "kp_onnx_test"

def run(test_files, modules=None, branch=True, html=True):
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    report, static, html_dir = ROOT / "coverage_report", ROOT / "coverage_report/static", ROOT / "coverage_report/html"
    static.mkdir(parents=True, exist_ok=True)
    data_file = static / ".coverage"

    print("=" * 80 + "\n开始运行测试并收集代码覆盖率...\n" + "=" * 80)
    cov = coverage.Coverage(source=[str(SRC)], branch=branch, data_file=str(data_file))
    cov.start()

    passed = True
    for tf in test_files:
        path = TEST / tf
        if not path.exists():
            print(f"警告: 测试文件不存在: {path}"); passed = False; continue
        print(f"\n运行测试: {tf}\n" + "-" * 80)
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit as e:
            if e.code not in (0, None): print(f"测试脚本退出码: {e.code}"); passed = False
        except Exception as e:
            print(f"执行 {tf} 时发生异常: {e}"); passed = False

    cov.stop(); cov.save()
    print("\n" + "=" * 80 + "\n代码覆盖率报告\n" + "=" * 80)

    inc = [str(SRC / f"{m if m.startswith('kop_') else f'kop_{m}'}.py") for m in modules] if modules else None
    if modules: print(f"\n只统计以下模块: {', '.join(modules)}")
    cov.report(include=inc, show_missing=True)

    if html:
        temp = report / "temp"
        shutil.rmtree(temp, ignore_errors=True)
        cov.html_report(directory=str(temp), include=inc)
        shutil.rmtree(html_dir, ignore_errors=True); html_dir.mkdir(parents=True)

        assets, htmls, renames = [], [], {}
        for f in temp.iterdir():
            if f.suffix == ".html" and f.name not in ("class_index.html", "function_index.html"):
                htmls.append(f)
                # 去掉 z_xxxxxxxx_ 前缀
                if f.name.startswith("z_") and f.name.count("_") >= 2:
                    new_name = f.name.split("_", 2)[2]
                    renames[f.name] = new_name
            elif f.suffix != ".html":
                assets.append(f.name); shutil.copy2(f, static / f.name)
        for f in htmls:
            c = f.read_text(encoding="utf-8")
            for a in assets: c = c.replace(f'"{a}"', f'"../static/{a}"')
            for old, new in renames.items(): c = c.replace(f'"{old}"', f'"{new}"')
            out_name = renames.get(f.name, f.name)
            (html_dir / out_name).write_text(c, encoding="utf-8")
        shutil.rmtree(temp)
        print(f"\nHTML 报告: {html_dir / 'index.html'}\n数据文件: {data_file}")
    return passed

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="代码覆盖率检查工具")
    p.add_argument("modules", nargs="*", help="模块名(如 kop_gru)")
    p.add_argument("--no-html", action="store_true")
    p.add_argument("--no-branch", action="store_true")
    a = p.parse_args()
    files = [f"{m if m.startswith('kop_') else f'kop_{m}'}_test.py" for m in a.modules] if a.modules else sorted(f.name for f in TEST.glob("kop_*_test.py"))
    if not a.modules: print(f"将检查所有测试文件: {len(files)} 个")
    sys.exit(0 if run(files, a.modules or None, not a.no_branch, not a.no_html) else 1)
