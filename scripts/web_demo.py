"""Interactive browser demo for the Valen Preview checkpoint."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from valen.data.schema import validate_record
from valen.evaluation.inference import predict
from valen.modeling.factory import build_compiler, build_model, get_backend, normalize_model_config
from valen.training.checkpoint import load_checkpoint
CHECKPOINT = ROOT / "models" / "Valen-Preview-0923"
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def parse_criteria(kind, text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if kind == "noul":
        return None
    if kind == "score":
        return lines
    criteria = {}
    for line in lines:
        name, separator, description = line.partition("|")
        name = name.strip()
        if not name or name in criteria:
            raise ValueError("候选名称不能为空或重复")
        criteria[name] = description.strip() if separator else name
    return criteria


def make_record(state, media_path, kind, instructions, criteria_text):
    state = state.strip()
    instructions = instructions.strip()
    if not state and not media_path:
        raise ValueError("请输入场景文字或上传图片、视频")
    question = {"type": kind, "instructions": instructions}
    criteria = parse_criteria(kind, criteria_text)
    if criteria is not None:
        question["criteria"] = criteria
    if media_path:
        path = Path(media_path).resolve(strict=True)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            media_type = "image_url"
        elif path.suffix.lower() in VIDEO_SUFFIXES:
            media_type = "video_url"
        else:
            raise ValueError("请上传常见格式的图片或视频文件")
        content = []
        if state:
            content.append({"type": "text", "text": state})
        content.append({"type": media_type, media_type: {"url": str(path)}})
        state = {"messages": [{"role": "user", "content": content}]}
    return {
        "group_id": "web-demo",
        "request": {"state": state, "questions": {"answer": question}},
    }


def format_answer(result):
    answer = result["answers"]["answer"]
    if answer["type"] == "choice":
        return f"**选择：{answer['choice']}** · 置信度：{answer['confidence']:.1%}"
    if answer["type"] == "noul":
        return f"**条件成立概率：{answer['noul']:.1%}**"
    return f"**评分：{answer['score']:.3f}** · 置信度：{answer['confidence']:.1%}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    config = json.loads((CHECKPOINT / "config.json").read_text(encoding="utf-8"))
    config = normalize_model_config(config)
    config["model_path"] = str(ROOT / config["model_path"])
    config["device"] = "cuda"
    model = build_model(config)
    load_checkpoint(CHECKPOINT, model)
    compiler = build_compiler(config, ROOT)
    backend = get_backend(config["architecture"])

    def infer(state, media_path, kind, instructions, criteria_text):
        try:
            record = make_record(state, media_path, kind, instructions, criteria_text)
            validate_record(record, backend.candidates)
            result = predict(model, compiler.compile(record))
            return format_answer(result), result
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="Valen 模型推理") as demo:
        gr.Markdown("# Valen 模型推理\n输入场景，任选一种决策任务；可上传一张图片或一个视频。模型输出候选概率，不生成自由文本。")
        with gr.Row():
            with gr.Column():
                state = gr.Textbox(label="场景描述", value="The traffic light is red.", lines=3)
                media = gr.File(label="图片或视频（可选）", file_types=["image", "video"], type="filepath")
                kind = gr.Radio(["choice", "noul", "score"], value="choice", label="任务类型")
                instructions = gr.Textbox(label="判断问题", value="What color is the traffic light?", lines=2)
                criteria = gr.Textbox(
                    label="候选项／评分等级（每行一项）",
                    value="red | Red light\ngreen | Green light\nunknown | Not visible",
                    lines=5,
                    info="Choice：名称 | 描述；Score：每行一个等级描述；Noul：无需填写。",
                )
                submit = gr.Button("运行推理", variant="primary")
            with gr.Column():
                summary = gr.Markdown()
                result = gr.JSON(label="完整结果")
        submit.click(infer, inputs=[state, media, kind, instructions, criteria], outputs=[summary, result], concurrency_limit=1)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=False, show_error=True
    )


if __name__ == "__main__":
    main()
