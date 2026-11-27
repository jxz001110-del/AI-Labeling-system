"""下载标注服务所需的开放词汇模型。

用法（在仓库根目录或任意目录）：
    python annotation_service/scripts/download_models.py

已存在的文件会自动跳过；支持通过 HF_ENDPOINT 使用 Hugging Face 镜像。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


SERVICE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = SERVICE_DIR / "models"
ULTRALYTICS_ASSETS = (
    "yolov8s-worldv2.pt",
    "yoloe-26s-seg.pt",
    "yoloe-26m-seg.pt",
    "yoloe-26l-seg.pt",
)
HF_MODELS = (
    ("IDEA-Research/grounding-dino-tiny", "grounding-dino-tiny"),
    ("google/owlv2-base-patch16-ensemble", "owlv2-base-patch16-ensemble"),
)
REQUIRED_LOCAL = (
    "visible_light.pt",
    "Object Detection.pt",
    "best.pt",
)


def _download_ultralytics() -> list[str]:
    from ultralytics.utils.downloads import attempt_download_asset

    failures: list[str] = []
    previous = Path.cwd()
    os.chdir(MODELS_DIR)
    try:
        for name in ULTRALYTICS_ASSETS:
            target = MODELS_DIR / name
            if target.exists():
                print(f"[skip] {name}")
                continue
            print(f"[download] {name}")
            try:
                downloaded = Path(attempt_download_asset(name))
                if downloaded.resolve() != target.resolve():
                    shutil.copy2(downloaded, target)
            except Exception as exc:  # 网络或镜像问题，继续处理其他文件
                failures.append(name)
                print(f"[fail] {name}: {exc}")

        text_encoder = SERVICE_DIR / "mobileclip2_b.ts"
        if text_encoder.exists():
            print("[skip] mobileclip2_b.ts")
        else:
            print("[download] mobileclip2_b.ts")
            try:
                downloaded = Path(attempt_download_asset("mobileclip2_b.ts"))
                shutil.copy2(downloaded, text_encoder)
            except Exception as exc:
                failures.append("mobileclip2_b.ts")
                print(f"[fail] mobileclip2_b.ts: {exc}")
    finally:
        os.chdir(previous)
    return failures


def _download_huggingface() -> list[str]:
    from huggingface_hub import snapshot_download

    failures: list[str] = []
    for repo_id, folder in HF_MODELS:
        target = MODELS_DIR / folder
        if (target / "model.safetensors").exists():
            print(f"[skip] {repo_id}")
            continue
        print(f"[download] {repo_id} -> {target}")
        try:
            snapshot_download(repo_id=repo_id, local_dir=str(target))
        except Exception as exc:
            failures.append(repo_id)
            print(f"[fail] {repo_id}: {exc}")
    return failures


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    missing_local = [
        name
        for name in REQUIRED_LOCAL
        if not (MODELS_DIR / name).exists() and not (SERVICE_DIR.parent / "models" / name).exists()
    ]
    if missing_local:
        print("[warn] 本机 7 类权重缺失：" + ", ".join(missing_local))
        print(f"       请把已有的权重文件放到：{MODELS_DIR}")
        print("       （这三个权重不在 Git 仓库中，需要从旧机器拷贝或页面上传）")

    failures = _download_ultralytics() + _download_huggingface()

    print("\n模型目录：", MODELS_DIR)
    if failures:
        print("[warn] 以下文件下载失败：" + ", ".join(failures))
        print("       可检查网络/代理，或从旧机器拷贝对应文件到相同位置。")
        return 1
    print("完成。重新打开巡检系统的「缺陷标注」页面即可看到模型。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
