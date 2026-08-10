from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
import sys
import json
import argparse


def generate_light_pptx(topic, points, page_num=5, data=None, save_path="light_presentation.pptx"):
    """
    生成极简轻量化 PPT 文件。

    参数:
        topic: PPT 主题（必填）
        points: 核心要点列表（必填，list[str]）
        page_num: 总页数（可选，默认 5，最大 8）
        data: 数据数组（可选，用于生成柱状图，至少 2 个元素）
        save_path: 保存路径（默认 light_presentation.pptx）

    返回:
        生成结果信息字符串
    """
    # 强制约束：页数 ≤ 8
    if page_num > 8:
        page_num = 8
    if page_num < 3:
        page_num = 3

    # 强制约束：每页要点 ≤ 3 条，自动截断
    if len(points) > 3:
        points = points[:3]

    prs = Presentation()

    # P1 封面
    cover = prs.slides.add_slide(prs.slide_layouts[0])
    cover.shapes.title.text = topic
    cover.placeholders[1].text = "极简轻量化简报"

    # P2 概述
    slide2 = prs.slides.add_slide(prs.slide_layouts[1])
    slide2.shapes.title.text = "内容概述"
    slide2.placeholders[1].text = f"本次简报核心：{topic}"

    # 正文分页（计算剩余可用页数）
    remaining = page_num - 2  # 减去封面和概述
    if data and len(data) >= 2:
        remaining -= 1  # 预留数据页
    remaining -= 1  # 预留总结页

    max_body_pages = max(1, remaining)
    body_points = points[:max_body_pages]

    for idx, point in enumerate(body_points):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = f"模块{idx + 1}"
        # 强制约束：单页文字 ≤ 60 字
        point_text = str(point)
        if len(point_text) > 60:
            point_text = point_text[:57] + "..."
        slide.placeholders[1].text = point_text

    # 数据页（可选）
    if data and len(data) >= 2:
        chart_slide = prs.slides.add_slide(prs.slide_layouts[1])
        chart_slide.shapes.title.text = "核心数据"
        chart_data = CategoryChartData()
        chart_data.categories = [str(i + 1) for i in range(len(data))]
        chart_data.add_series("数值", [float(x) for x in data])
        chart_slide.shapes.add_chart(
            XL_CHART_TYPE.COLUMN_CLUSTERED, 0, 2000000, 6000000, 4000000, chart_data
        )

    # 总结页
    end_slide = prs.slides.add_slide(prs.slide_layouts[1])
    end_slide.shapes.title.text = "总结与计划"
    end_slide.placeholders[1].text = "1. 完成本次核心内容\n2. 后续持续优化落地"

    prs.save(save_path)
    return f"PPT已生成：{save_path}（共{len(prs.slides)}页）"


def _parse_args():
    parser = argparse.ArgumentParser(description="ynlczf-pptx-generator: 超轻量化PPT生成工具")
    parser.add_argument("--topic", required=True, help="PPT 主题")
    parser.add_argument("--points", required=True, help="核心要点，逗号分隔（例如：要点1,要点2,要点3）")
    parser.add_argument("--page-num", type=int, default=5, help="总页数（默认5，最大8）")
    parser.add_argument("--data", default=None, help="可选数据数组，逗号分隔数字（例如：10,25,30,15）")
    parser.add_argument("--output", default="light_presentation.pptx", help="输出文件路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")
    return parser.parse_args()


def main():
    args = _parse_args()

    points = [p.strip() for p in args.points.split(",") if p.strip()]
    if not points:
        print("错误：points 不能为空", file=sys.stderr)
        sys.exit(1)

    data = None
    if args.data:
        try:
            data = [float(x.strip()) for x in args.data.split(",") if x.strip()]
        except ValueError:
            print("警告：data 参数解析失败，跳过数据页生成", file=sys.stderr)
            data = None

    try:
        result = generate_light_pptx(
            topic=args.topic,
            points=points,
            page_num=args.page_num,
            data=data,
            save_path=args.output,
        )
        if args.json:
            print(json.dumps({"status": "ok", "message": result, "output": args.output}, ensure_ascii=False))
        else:
            print(result)
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False))
        else:
            print(f"生成失败：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
