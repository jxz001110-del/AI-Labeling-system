"""缺陷标注试用系统（独立应用，不依赖 Docker / CVAT / PostgreSQL）。

流程：批量导入图片 → 选择 AI 模型 → 配置缺陷类型 → AI 预标注出框 →
人工逐框确认类型 → 导出标注结果 → 汇总本批次统计。
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_DIR = BASE_DIR / "models"
EXPORT_DIR = BASE_DIR / "exports"
STATIC_DIR = BASE_DIR / "static"
DB_PATH = DATA_DIR / "labeler.db"

for directory in (DATA_DIR, UPLOAD_DIR, MODEL_DIR, EXPORT_DIR, STATIC_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# 本机已有的可用权重（只登记存在文件，不下载、不伪造）
# 优先使用标注服务自己的 models/，兼容作为宿主仓库子目录时的 ../models。
REPO_MODEL_DIR = BASE_DIR.parent / "models"


def _local_weight(name: str) -> str:
    for base in (MODEL_DIR, REPO_MODEL_DIR):
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    return str(MODEL_DIR / name)


MODEL_CANDIDATES: list[tuple[str, str]] = [
    ("可见光缺陷分割（7类）", _local_weight("visible_light.pt")),
    ("中文缺陷检测（7类）", _local_weight("Object Detection.pt")),
    ("裂缝检测（单类）", _local_weight("best.pt")),
    ("YOLO-World 开放词汇（通用）", str(MODEL_DIR / "yolov8s-worldv2.pt")),
    ("YOLOE-26S 开放词汇（候选框）", str(MODEL_DIR / "yoloe-26s-seg.pt")),
    ("YOLOE-26M 开放词汇（候选框）", str(MODEL_DIR / "yoloe-26m-seg.pt")),
    ("YOLOE-26L 开放词汇（候选框）", str(MODEL_DIR / "yoloe-26l-seg.pt")),
    ("Grounding DINO Tiny（开放词汇）", str(MODEL_DIR / "grounding-dino-tiny")),
    ("OWLv2 Base Ensemble（开放词汇）", str(MODEL_DIR / "owlv2-base-patch16-ensemble")),
]

SEED_DEFECT_TYPES = [
    ("crack", "裂缝", "#ff5c5c"),
    ("exposed_rebar", "露筋", "#ff9f43"),
    ("peeling", "脱落", "#f5d90a"),
    ("corner_damage", "缺棱断角", "#4f8cff"),
    ("rebar_corrosion", "露筋锈蚀", "#b36bff"),
    ("fracture", "断裂", "#22c55e"),
    ("honeycombing", "蜂窝麻面", "#00c8c8"),
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_IMAGE_BYTES = 80 * 1024 * 1024
MAX_MODEL_BYTES = 2 * 1024 * 1024 * 1024

app = FastAPI(title="缺陷标注试用系统", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

MODEL_CACHE: dict[int, tuple[Any, dict[int, str], str]] = {}
CACHE_LOCK = threading.Lock()
RUNNING: dict[int, dict[str, Any]] = {}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_columns(conn: sqlite3.Connection) -> None:
    """为早期版本创建的数据库补齐标注员相关字段。"""

    def columns(table: str) -> set[str]:
        return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}

    box_columns = columns("boxes")
    if "reviewed_by" not in box_columns:
        conn.execute("ALTER TABLE boxes ADD COLUMN reviewed_by INTEGER")
    if "reviewed_at" not in box_columns:
        conn.execute("ALTER TABLE boxes ADD COLUMN reviewed_at TEXT")
    if "source" not in box_columns:
        conn.execute("ALTER TABLE boxes ADD COLUMN source TEXT NOT NULL DEFAULT 'AI'")
    image_columns = columns("images")
    if "done_by" not in image_columns:
        conn.execute("ALTER TABLE images ADD COLUMN done_by INTEGER")
    if "done_at" not in image_columns:
        conn.execute("ALTER TABLE images ADD COLUMN done_at TEXT")
    batch_columns = columns("batches")
    if "mode" not in batch_columns:
        # 历史批次沿用“类型建议”，新批次默认“只出候选框”
        conn.execute("ALTER TABLE batches ADD COLUMN mode TEXT NOT NULL DEFAULT 'SUGGEST'")
    if "prompt" not in batch_columns:
        conn.execute("ALTER TABLE batches ADD COLUMN prompt TEXT")


def init_db() -> None:
    with db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                task TEXT NOT NULL,
                classes_json TEXT NOT NULL DEFAULT '[]',
                size INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'auto',
                error TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS defect_types (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                color TEXT NOT NULL DEFAULT '#4f8cff',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS annotators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                code TEXT,
                note TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                model_id INTEGER NOT NULL REFERENCES models(id),
                conf REAL NOT NULL DEFAULT 0.25,
                iou REAL NOT NULL DEFAULT 0.45,
                status TEXT NOT NULL DEFAULT 'CREATED',
                total INTEGER NOT NULL DEFAULT 0,
                processed INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT
            );
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                path TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                status TEXT NOT NULL DEFAULT 'UPLOADED',
                error TEXT,
                done_by INTEGER,
                done_at TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS boxes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                score REAL,
                ai_label TEXT,
                defect_type_id INTEGER REFERENCES defect_types(id),
                decision TEXT NOT NULL DEFAULT 'PENDING',
                source TEXT NOT NULL DEFAULT 'AI',
                reviewed_by INTEGER,
                reviewed_at TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_images_batch ON images(batch_id);
            CREATE INDEX IF NOT EXISTS idx_boxes_image ON boxes(image_id);
            """
        )
        ensure_columns(conn)
        count = conn.execute("SELECT COUNT(*) AS c FROM defect_types").fetchone()["c"]
        if count == 0:
            conn.executemany(
                "INSERT INTO defect_types(code,name,color,active,created_at) VALUES(?,?,?,1,?)",
                [(code, name, color, now_text()) for code, name, color in SEED_DEFECT_TYPES],
            )
        # 服务重启后把残留的运行状态标为中断，避免误以为仍在检测
        conn.execute(
            "UPDATE batches SET status='INTERRUPTED', error=COALESCE(error,'服务重启导致标注中断') "
            "WHERE status='DETECTING'"
        )


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def model_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = row_to_dict(row)
    try:
        data["classes"] = json.loads(data.pop("classes_json") or "[]")
    except json.JSONDecodeError:
        data["classes"] = []
    data["ready"] = data["task"] not in ("PROBING", "ERROR")
    return data


def is_open_vocab_path(path_text: str) -> bool:
    lowered = path_text.lower()
    return "world" in lowered or "yoloe" in lowered or "grounding" in lowered or "owlv2" in lowered


class _GroundingBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class _GroundingResult:
    def __init__(self, boxes):
        self.boxes = boxes


class GroundingDinoAdapter:
    """将 Grounding DINO 适配成与 Ultralytics 一致的 predict 返回结构。"""

    task = "detect"

    def __init__(self, path: Path):
        self.path = Path(path)
        self._prompts = ["defect", "damage", "crack", "spalling", "corrosion"]
        self.names = {index: name for index, name in enumerate(self._prompts)}
        self._processor = None
        self._model = None
        self._device = None

    def set_classes(self, classes) -> None:
        prompts = [str(item).strip() for item in classes if str(item).strip()]
        if prompts:
            self._prompts = prompts
            self.names = {index: name for index, name in enumerate(prompts)}

    def _ensure_loaded(self, device: str | None) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if self._model is None or self._processor is None:
            self._processor = AutoProcessor.from_pretrained(str(self.path))
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(str(self.path))
        target = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if self._device != target:
            self._model.to(target)
            self._device = target
        self._model.eval()

    def predict(self, source, conf: float = 0.25, iou: float = 0.45, verbose: bool = False,
                device: str | None = None):
        import torch
        from PIL import Image

        self._ensure_loaded(device)
        image = Image.fromarray(source[:, :, ::-1].copy())
        text = ". ".join(self._prompts).strip()
        if not text.endswith("."):
            text += "."
        inputs = self._processor(images=image, text=text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        processed = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=float(conf),
            text_threshold=float(conf),
            target_sizes=[image.size[::-1]],
        )[0]
        raw_boxes = processed.get("boxes", [])
        raw_scores = processed.get("scores", [])
        raw_labels = processed.get("labels")
        if raw_labels is None:
            raw_labels = processed.get("text_labels", [])
        if torch.is_tensor(raw_boxes):
            boxes = raw_boxes.float().reshape(-1, 4).cpu()
        elif len(raw_boxes):
            boxes = torch.stack([torch.as_tensor(item, dtype=torch.float32) for item in raw_boxes])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        if torch.is_tensor(raw_scores):
            scores = raw_scores.float().reshape(-1).cpu()
        elif len(raw_scores):
            scores = torch.as_tensor(raw_scores, dtype=torch.float32)
        else:
            scores = torch.zeros((0,), dtype=torch.float32)
        lowered = [prompt.lower() for prompt in self._prompts]
        classes: list[int] = []
        for index in range(boxes.shape[0]):
            label = str(raw_labels[index]).strip().lower() if index < len(raw_labels) else ""
            matched = 0
            for position, prompt in enumerate(lowered):
                if label and (label == prompt or label in prompt or prompt in label):
                    matched = position
                    break
            classes.append(matched)
        class_tensor = torch.as_tensor(classes, dtype=torch.float32)
        if boxes.shape[0] > 0:
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=image.width)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=image.height)
            from torchvision.ops import nms

            keep = nms(boxes, scores, float(iou))
            boxes = boxes[keep]
            scores = scores[keep]
            class_tensor = class_tensor[keep]
        return [_GroundingResult(_GroundingBoxes(boxes, scores, class_tensor))]


class OwlV2Adapter:
    """将 OWLv2 适配成与 Ultralytics 一致的 predict 返回结构。"""

    task = "detect"

    def __init__(self, path: Path):
        self.path = Path(path)
        self._prompts = ["defect", "damage", "crack", "spalling", "corrosion"]
        self.names = {index: name for index, name in enumerate(self._prompts)}
        self._processor = None
        self._model = None
        self._device = None

    def set_classes(self, classes) -> None:
        prompts = [str(item).strip() for item in classes if str(item).strip()]
        if prompts:
            self._prompts = prompts
            self.names = {index: name for index, name in enumerate(prompts)}

    def _ensure_loaded(self, device: str | None) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if self._model is None or self._processor is None:
            self._processor = AutoProcessor.from_pretrained(str(self.path))
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(str(self.path))
        target = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if self._device != target:
            self._model.to(target)
            self._device = target
        self._model.eval()

    def predict(self, source, conf: float = 0.25, iou: float = 0.45, verbose: bool = False,
                device: str | None = None):
        import torch
        from PIL import Image

        self._ensure_loaded(device)
        image = Image.fromarray(source[:, :, ::-1].copy())
        texts = [[f"a photo of a {prompt}" for prompt in self._prompts]]
        inputs = self._processor(text=texts, images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        processed = self._processor.post_process_grounded_object_detection(
            outputs,
            threshold=float(conf),
            target_sizes=[image.size[::-1]],
            text_labels=texts,
        )[0]
        raw_boxes = processed.get("boxes", [])
        raw_scores = processed.get("scores", [])
        raw_labels = processed.get("labels", [])
        if torch.is_tensor(raw_boxes):
            boxes = raw_boxes.float().reshape(-1, 4).cpu()
        elif len(raw_boxes):
            boxes = torch.stack([torch.as_tensor(item, dtype=torch.float32) for item in raw_boxes])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        if torch.is_tensor(raw_scores):
            scores = raw_scores.float().reshape(-1).cpu()
        elif len(raw_scores):
            scores = torch.as_tensor(raw_scores, dtype=torch.float32)
        else:
            scores = torch.zeros((0,), dtype=torch.float32)
        if torch.is_tensor(raw_labels):
            classes = raw_labels.long().reshape(-1).cpu()
        else:
            classes = torch.as_tensor([int(item) for item in raw_labels], dtype=torch.long)
        if boxes.shape[0] > 0:
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=image.width)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=image.height)
            from torchvision.ops import nms

            keep = nms(boxes, scores, float(iou))
            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]
        return [_GroundingResult(_GroundingBoxes(boxes, scores, classes.float()))]


def build_model_loader(path: Path):
    if "owlv2" in path.name.lower():
        return OwlV2Adapter(path)
    if path.is_dir() or "grounding-dino" in path.name.lower():
        return GroundingDinoAdapter(path)
    from ultralytics import YOLO

    if "yoloe" in path.name.lower():
        from ultralytics import YOLOE

        return YOLOE(str(path))
    return YOLO(str(path))


def probe_model_file(path: Path) -> tuple[str, list[dict[str, Any]]]:
    model = build_model_loader(path)
    task = str(getattr(model, "task", "") or "detect")
    names = getattr(model, "names", {}) or {}
    classes = [{"id": int(key), "name": str(value)} for key, value in sorted(names.items(), key=lambda kv: int(kv[0]))]
    return task, classes


def device_info() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            return {"device": "cuda:0", "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
        return {"device": "cpu", "gpu": None, "cuda": None}
    except Exception as exc:  # noqa: BLE001
        return {"device": "cpu", "gpu": None, "error": str(exc)}


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def ensure_candidate_models() -> None:
    with db() as conn:
        for name, path_text in MODEL_CANDIDATES:
            path = Path(path_text)
            if not path.exists():
                continue
            exists = conn.execute("SELECT id FROM models WHERE path=?", (str(path),)).fetchone()
            if exists:
                continue
            conn.execute(
                "INSERT INTO models(name,path,task,classes_json,size,source,created_at) VALUES(?,?,?,?,?,?,?)",
                (name, str(path), "PROBING", "[]", directory_size(path), "auto", now_text()),
            )
    start_probe_thread()


def start_probe_thread() -> None:
    thread = threading.Thread(target=probe_pending_models, name="model-probe", daemon=True)
    thread.start()


def probe_pending_models() -> None:
    with db() as conn:
        rows = conn.execute("SELECT * FROM models WHERE task='PROBING'").fetchall()
    for row in rows:
        try:
            task, classes = probe_model_file(Path(row["path"]))
            with db() as conn:
                conn.execute(
                    "UPDATE models SET task=?, classes_json=?, error=NULL WHERE id=?",
                    (task, json.dumps(classes, ensure_ascii=False), row["id"]),
                )
            print(f"[model] 探测成功 {row['name']} task={task} classes={len(classes)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            with db() as conn:
                conn.execute("UPDATE models SET task='ERROR', error=? WHERE id=?", (str(exc)[:500], row["id"]))
            print(f"[model] 探测失败 {row['name']}: {exc}", flush=True)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    ensure_candidate_models()
    print("[startup] 缺陷标注试用系统已启动", flush=True)


@app.get("/api/health")
def health() -> dict[str, Any]:
    info = device_info()
    return {"ok": True, "version": app.version, "time": now_text(), **info}


# ---------------------------------------------------------------- 模型管理
@app.get("/api/models")
def list_models() -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM models ORDER BY id").fetchall()
    return [model_to_dict(row) for row in rows]


@app.post("/api/models/upload")
def upload_model(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".pt":
        raise HTTPException(status_code=400, detail="仅支持 .pt 权重文件")
    target = MODEL_DIR / f"{uuid.uuid4().hex[:8]}_{Path(file.filename or 'model.pt').name}"
    size = 0
    with target.open("wb") as handle:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_MODEL_BYTES:
                handle.close()
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="权重文件超过 2GB 限制")
            handle.write(chunk)
    try:
        task, classes = probe_model_file(target)
    except Exception as exc:  # noqa: BLE001
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"无法识别该权重：{exc}") from exc
    with db() as conn:
        cursor = conn.execute(
            "INSERT INTO models(name,path,task,classes_json,size,source,created_at) VALUES(?,?,?,?,?,?,?)",
            (Path(file.filename or target.name).stem, str(target), task,
             json.dumps(classes, ensure_ascii=False), size, "upload", now_text()),
        )
        model_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
    return model_to_dict(row)


# ---------------------------------------------------------------- 缺陷类型
@app.get("/api/defect-types")
def list_defect_types(include_inactive: bool = True) -> list[dict[str, Any]]:
    sql = "SELECT * FROM defect_types"
    if not include_inactive:
        sql += " WHERE active=1"
    sql += " ORDER BY id"
    with db() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]


class DefectTypeCreate(BaseModel):
    code: str
    name: str
    color: str = "#4f8cff"


class DefectTypePatch(BaseModel):
    code: str | None = None
    name: str | None = None
    color: str | None = None
    active: bool | None = None


def clean_code(code: str) -> str:
    value = code.strip()
    if not value:
        raise HTTPException(status_code=400, detail="缺陷编码不能为空")
    return value


@app.post("/api/defect-types")
def create_defect_type(payload: DefectTypeCreate) -> dict[str, Any]:
    code = clean_code(payload.code)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="缺陷名称不能为空")
    with db() as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO defect_types(code,name,color,active,created_at) VALUES(?,?,?,1,?)",
                (code, name, payload.color, now_text()),
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail=f"缺陷编码 {code} 已存在") from exc
        row = conn.execute("SELECT * FROM defect_types WHERE id=?", (cursor.lastrowid,)).fetchone()
    return dict(row)


@app.patch("/api/defect-types/{type_id}")
def update_defect_type(type_id: int, payload: DefectTypePatch) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT * FROM defect_types WHERE id=?", (type_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="缺陷类型不存在")
        updates: dict[str, Any] = {}
        if payload.code is not None:
            updates["code"] = clean_code(payload.code)
        if payload.name is not None:
            name = payload.name.strip()
            if not name:
                raise HTTPException(status_code=400, detail="缺陷名称不能为空")
            updates["name"] = name
        if payload.color is not None:
            updates["color"] = payload.color
        if payload.active is not None:
            updates["active"] = 1 if payload.active else 0
        if updates:
            try:
                conn.execute(
                    f"UPDATE defect_types SET {', '.join(f'{key}=?' for key in updates)} WHERE id=?",
                    (*updates.values(), type_id),
                )
            except sqlite3.IntegrityError as exc:
                raise HTTPException(status_code=409, detail="缺陷编码重复") from exc
        row = conn.execute("SELECT * FROM defect_types WHERE id=?", (type_id,)).fetchone()
    return dict(row)


# ---------------------------------------------------------------- 标注员
class AnnotatorCreate(BaseModel):
    name: str
    code: str | None = None
    note: str | None = None


class AnnotatorPatch(BaseModel):
    name: str | None = None
    code: str | None = None
    note: str | None = None
    active: bool | None = None


def require_annotator(conn: sqlite3.Connection, annotator_id: int | None) -> sqlite3.Row:
    if not annotator_id:
        raise HTTPException(status_code=400, detail="请先选择人工标注员")
    row = conn.execute("SELECT * FROM annotators WHERE id=?", (annotator_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="标注员不存在")
    if not row["active"]:
        raise HTTPException(status_code=400, detail=f"标注员 {row['name']} 已停用")
    return row


@app.get("/api/annotators")
def list_annotators() -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT a.*,
              COALESCE((SELECT COUNT(*) FROM boxes b WHERE b.reviewed_by=a.id AND b.decision='ACCEPTED'),0) AS accepted,
              COALESCE((SELECT COUNT(*) FROM boxes b WHERE b.reviewed_by=a.id AND b.decision='REJECTED'),0) AS rejected,
              COALESCE((SELECT COUNT(*) FROM images i WHERE i.done_by=a.id),0) AS done_images
            FROM annotators a ORDER BY a.active DESC, a.id
            """
        ).fetchall()
    return [row_to_dict(row) for row in rows]


@app.post("/api/annotators")
def create_annotator(payload: AnnotatorCreate) -> dict[str, Any]:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="标注员姓名不能为空")
    with db() as conn:
        cursor = conn.execute(
            "INSERT INTO annotators(name,code,note,active,created_at) VALUES(?,?,?,1,?)",
            (name, (payload.code or "").strip() or None, (payload.note or "").strip() or None, now_text()),
        )
        row = conn.execute("SELECT * FROM annotators WHERE id=?", (cursor.lastrowid,)).fetchone()
    return row_to_dict(row)


@app.patch("/api/annotators/{annotator_id}")
def update_annotator(annotator_id: int, payload: AnnotatorPatch) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT * FROM annotators WHERE id=?", (annotator_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="标注员不存在")
        updates: dict[str, Any] = {}
        if payload.name is not None:
            name = payload.name.strip()
            if not name:
                raise HTTPException(status_code=400, detail="标注员姓名不能为空")
            updates["name"] = name
        if payload.code is not None:
            updates["code"] = payload.code.strip() or None
        if payload.note is not None:
            updates["note"] = payload.note.strip() or None
        if payload.active is not None:
            updates["active"] = 1 if payload.active else 0
        if updates:
            conn.execute(
                f"UPDATE annotators SET {', '.join(f'{key}=?' for key in updates)} WHERE id=?",
                (*updates.values(), annotator_id),
            )
        row = conn.execute("SELECT * FROM annotators WHERE id=?", (annotator_id,)).fetchone()
    return row_to_dict(row)


# ---------------------------------------------------------------- 批次与图片
class BatchCreate(BaseModel):
    name: str
    model_id: int
    conf: float = 0.25
    iou: float = 0.45
    mode: str = "BOX_ONLY"
    prompt: str | None = None


def batch_summary(conn: sqlite3.Connection, batch_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT b.*, m.name AS model_name, m.task AS model_task FROM batches b "
        "JOIN models m ON m.id=b.model_id WHERE b.id=?",
        (batch_id,),
    ).fetchone()
    if not row:
        return None
    data = row_to_dict(row)
    data["image_count"] = conn.execute(
        "SELECT COUNT(*) AS c FROM images WHERE batch_id=?", (batch_id,)
    ).fetchone()["c"]
    data["detected_count"] = conn.execute(
        "SELECT COUNT(*) AS c FROM images WHERE batch_id=? AND status IN ('DETECTED','REVIEWING','DONE')",
        (batch_id,),
    ).fetchone()["c"]
    data["done_count"] = conn.execute(
        "SELECT COUNT(*) AS c FROM images WHERE batch_id=? AND status='DONE'", (batch_id,)
    ).fetchone()["c"]
    data["box_count"] = conn.execute(
        "SELECT COUNT(*) AS c FROM boxes b JOIN images i ON i.id=b.image_id WHERE i.batch_id=?",
        (batch_id,),
    ).fetchone()["c"]
    data["pending_boxes"] = conn.execute(
        "SELECT COUNT(*) AS c FROM boxes b JOIN images i ON i.id=b.image_id "
        "WHERE i.batch_id=? AND b.decision='PENDING'",
        (batch_id,),
    ).fetchone()["c"]
    return data


@app.post("/api/batches")
def create_batch(payload: BatchCreate) -> dict[str, Any]:
    name = payload.name.strip() or datetime.now().strftime("批次-%Y%m%d-%H%M")
    mode = payload.mode.upper()
    if mode not in ("BOX_ONLY", "SUGGEST"):
        raise HTTPException(status_code=400, detail="mode 只能是 BOX_ONLY 或 SUGGEST")
    prompt = (payload.prompt or "").strip()
    with db() as conn:
        model = conn.execute("SELECT * FROM models WHERE id=?", (payload.model_id,)).fetchone()
        if not model:
            raise HTTPException(status_code=404, detail="所选模型不存在")
        if model["task"] in ("PROBING", "ERROR"):
            raise HTTPException(status_code=400, detail="所选模型尚未就绪，请稍后重试或更换模型")
        if is_open_vocab_path(model["path"]) and not prompt:
            raise HTTPException(status_code=400, detail="开放词汇模型需要填写 AI 提示词，例如 defect, damage, crack")
        cursor = conn.execute(
            "INSERT INTO batches(name,model_id,conf,iou,status,total,processed,mode,prompt,created_at) "
            "VALUES(?,?,?,?,?,0,0,?,?,?)",
            (name, payload.model_id, round(payload.conf, 4), round(payload.iou, 4), "CREATED",
             mode, prompt or None, now_text()),
        )
        batch_id = cursor.lastrowid
        batch = batch_summary(conn, batch_id)
    return batch


@app.get("/api/batches")
def list_batches() -> list[dict[str, Any]]:
    with db() as conn:
        ids = [row["id"] for row in conn.execute("SELECT id FROM batches ORDER BY id DESC").fetchall()]
        return [batch_summary(conn, batch_id) for batch_id in ids]


@app.get("/api/batches/{batch_id}")
def get_batch(batch_id: int) -> dict[str, Any]:
    with db() as conn:
        batch = batch_summary(conn, batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail="批次不存在")
        model = conn.execute("SELECT * FROM models WHERE id=?", (batch["model_id"],)).fetchone()
        image_rows = conn.execute(
            """
            SELECT i.*, COUNT(b.id) AS box_count,
                   COALESCE(SUM(CASE WHEN b.decision='PENDING' THEN 1 ELSE 0 END),0) AS pending,
                   COALESCE(SUM(CASE WHEN b.decision='ACCEPTED' THEN 1 ELSE 0 END),0) AS accepted,
                   COALESCE(SUM(CASE WHEN b.decision='REJECTED' THEN 1 ELSE 0 END),0) AS rejected
            FROM images i LEFT JOIN boxes b ON b.image_id=i.id
            WHERE i.batch_id=? GROUP BY i.id ORDER BY i.id
            """,
            (batch_id,),
        ).fetchall()
    images = []
    for row in image_rows:
        item = row_to_dict(row)
        item["url"] = f"/api/images/{item['id']}/file"
        images.append(item)
    return {"batch": batch, "model": model_to_dict(model), "images": images}


@app.delete("/api/batches/{batch_id}")
def delete_batch(batch_id: int) -> dict[str, Any]:
    with db() as conn:
        batch = conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
        if not batch:
            raise HTTPException(status_code=404, detail="批次不存在")
        if batch["status"] == "DETECTING":
            raise HTTPException(status_code=409, detail="批次正在标注中，无法删除")
        conn.execute("DELETE FROM batches WHERE id=?", (batch_id,))
    shutil.rmtree(UPLOAD_DIR / f"batch_{batch_id}", ignore_errors=True)
    return {"ok": True}


@app.post("/api/batches/{batch_id}/images")
def upload_images(batch_id: int, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    with db() as conn:
        batch = conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
        if not batch:
            raise HTTPException(status_code=404, detail="批次不存在")
        if batch["status"] == "DETECTING":
            raise HTTPException(status_code=409, detail="批次正在标注中，暂不能上传")
    target_dir = UPLOAD_DIR / f"batch_{batch_id}"
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    rejected: list[str] = []
    for file in files:
        name = Path(file.filename or "").name
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            rejected.append(f"{name}（不支持的格式）")
            continue
        target = target_dir / f"{uuid.uuid4().hex[:8]}_{name}"
        size = 0
        try:
            with target.open("wb") as handle:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_IMAGE_BYTES:
                        raise ValueError("图片超过 80MB")
                    handle.write(chunk)
            with Image.open(target) as image:
                width, height = image.size
                image.verify()
        except Exception as exc:  # noqa: BLE001
            target.unlink(missing_ok=True)
            rejected.append(f"{name}（{exc}）")
            continue
        with db() as conn:
            cursor = conn.execute(
                "INSERT INTO images(batch_id,filename,path,width,height,status,created_at) "
                "VALUES(?,?,?,?,?,'UPLOADED',?)",
                (batch_id, name, str(target), width, height, now_text()),
            )
            saved.append({"id": cursor.lastrowid, "filename": name, "width": width, "height": height})
    with db() as conn:
        conn.execute("UPDATE batches SET total=(SELECT COUNT(*) FROM images WHERE batch_id=?) WHERE id=?",
                     (batch_id, batch_id))
        refresh_batch_status(conn, batch_id)
        batch = batch_summary(conn, batch_id)
    return {"saved": saved, "rejected": rejected, "batch": batch}


# ---------------------------------------------------------------- 单图与框
def image_status_from_boxes(pending: int, total: int, current: str) -> str:
    if current in ("UPLOADED", "ERROR"):
        return current
    if total == 0:
        return current if current == "DONE" else "DETECTED"
    return "DONE" if pending == 0 else "REVIEWING"


def refresh_image_status(conn: sqlite3.Connection, image_id: int) -> str:
    row = conn.execute(
        "SELECT i.status, COUNT(b.id) AS total, "
        "COALESCE(SUM(CASE WHEN b.decision='PENDING' THEN 1 ELSE 0 END),0) AS pending "
        "FROM images i LEFT JOIN boxes b ON b.image_id=i.id WHERE i.id=? GROUP BY i.id",
        (image_id,),
    ).fetchone()
    status = image_status_from_boxes(row["pending"], row["total"], row["status"])
    if status == "DONE":
        conn.execute("UPDATE images SET status=? WHERE id=?", (status, image_id))
    else:
        conn.execute("UPDATE images SET status=?, done_by=NULL, done_at=NULL WHERE id=?", (status, image_id))
    return status


def refresh_batch_status(conn: sqlite3.Connection, batch_id: int) -> None:
    batch = conn.execute("SELECT status FROM batches WHERE id=?", (batch_id,)).fetchone()
    if not batch or batch["status"] in ("DETECTING", "INTERRUPTED"):
        return
    stats = conn.execute(
        """
        SELECT COUNT(i.id) AS total,
               COALESCE(SUM(CASE WHEN i.status='DONE' THEN 1 ELSE 0 END),0) AS done,
               COALESCE(SUM(CASE WHEN i.status IN ('DETECTED','REVIEWING','DONE') THEN 1 ELSE 0 END),0) AS detected,
               (SELECT COUNT(*) FROM boxes b JOIN images i2 ON i2.id=b.image_id
                WHERE i2.batch_id=? AND b.decision<>'PENDING') AS decided
        FROM images i WHERE i.batch_id=?
        """,
        (batch_id, batch_id),
    ).fetchone()
    if stats["total"] == 0:
        status = "CREATED"
    elif stats["done"] >= stats["total"]:
        status = "COMPLETED"
    elif stats["decided"] > 0:
        status = "REVIEWING"
    elif stats["detected"] > 0:
        status = "DETECTED"
    else:
        status = "UPLOADED"
    conn.execute("UPDATE batches SET status=? WHERE id=?", (status, batch_id))


def image_detail(conn: sqlite3.Connection, image_id: int) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="图片不存在")
    data = row_to_dict(row)
    data["url"] = f"/api/images/{image_id}/file"
    boxes = conn.execute("SELECT * FROM boxes WHERE image_id=? ORDER BY id", (image_id,)).fetchall()
    types = {item["id"]: dict(item) for item in conn.execute("SELECT * FROM defect_types").fetchall()}
    annotators = {item["id"]: dict(item) for item in conn.execute("SELECT id,name,code FROM annotators").fetchall()}
    result = []
    for box in boxes:
        item = row_to_dict(box)
        type_row = types.get(item["defect_type_id"])
        item["defect_type"] = type_row
        item["reviewer"] = annotators.get(item["reviewed_by"])
        result.append(item)
    data["boxes"] = result
    data["done_by_name"] = (annotators.get(data.get("done_by")) or {}).get("name")
    data["pending"] = sum(1 for item in result if item["decision"] == "PENDING")
    data["accepted"] = sum(1 for item in result if item["decision"] == "ACCEPTED")
    data["rejected"] = sum(1 for item in result if item["decision"] == "REJECTED")
    return data


@app.get("/api/images/{image_id}")
def get_image(image_id: int) -> dict[str, Any]:
    with db() as conn:
        return image_detail(conn, image_id)


@app.get("/api/images/{image_id}/file")
def get_image_file(image_id: int):
    with db() as conn:
        row = conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="图片不存在")
    path = Path(row["path"])
    if not path.exists() or not path.resolve().is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(status_code=404, detail="图片文件已丢失")
    return FileResponse(path)


class BoxCreate(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    defect_type_id: int | None = None


@app.post("/api/images/{image_id}/boxes")
def create_manual_box(image_id: int, payload: BoxCreate) -> dict[str, Any]:
    with db() as conn:
        image = conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        if not image:
            raise HTTPException(status_code=404, detail="图片不存在")
        width = float(image["width"] or 0)
        height = float(image["height"] or 0)
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="图片尺寸未知，无法新增框")
        x1, x2 = sorted((float(payload.x1), float(payload.x2)))
        y1, y2 = sorted((float(payload.y1), float(payload.y2)))
        x1 = max(0.0, min(x1, width))
        x2 = max(0.0, min(x2, width))
        y1 = max(0.0, min(y1, height))
        y2 = max(0.0, min(y2, height))
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise HTTPException(status_code=400, detail="框范围太小，请重新画框")
        if payload.defect_type_id is not None:
            exists = conn.execute("SELECT 1 FROM defect_types WHERE id=?", (payload.defect_type_id,)).fetchone()
            if not exists:
                raise HTTPException(status_code=404, detail="缺陷类型不存在")
        cursor = conn.execute(
            "INSERT INTO boxes(image_id,x1,y1,x2,y2,score,ai_label,defect_type_id,decision,source,created_at) "
            "VALUES(?,?,?,?,?,NULL,NULL,?,'PENDING','MANUAL',?)",
            (image_id, round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
             payload.defect_type_id, now_text()),
        )
        box_id = int(cursor.lastrowid)
        refresh_image_status(conn, image_id)
        refresh_batch_status(conn, image["batch_id"])
        detail = image_detail(conn, image_id)
    return {"box_id": box_id, "image": detail}


class BoxPatch(BaseModel):
    decision: str | None = None
    defect_type_id: int | None = None
    annotator_id: int | None = None
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None


def apply_box_patch(conn: sqlite3.Connection, box_id: int, payload: BoxPatch) -> dict[str, Any]:
    box = conn.execute("SELECT * FROM boxes WHERE id=?", (box_id,)).fetchone()
    if not box:
        raise HTTPException(status_code=404, detail="标注框不存在")
    coords = (payload.x1, payload.y1, payload.x2, payload.y2)
    if any(value is not None for value in coords):
        if any(value is None for value in coords):
            raise HTTPException(status_code=400, detail="调整框范围需要同时提供 x1/y1/x2/y2")
        if box["decision"] != "PENDING":
            raise HTTPException(status_code=400, detail="只有待确认的框可以调整范围")
        image_row = conn.execute("SELECT width,height FROM images WHERE id=?", (box["image_id"],)).fetchone()
        width = float(image_row["width"] or 0)
        height = float(image_row["height"] or 0)
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="图片尺寸未知，无法调整范围")
        x1, x2 = sorted((float(payload.x1), float(payload.x2)))
        y1, y2 = sorted((float(payload.y1), float(payload.y2)))
        x1 = max(0.0, min(x1, width))
        x2 = max(0.0, min(x2, width))
        y1 = max(0.0, min(y1, height))
        y2 = max(0.0, min(y2, height))
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise HTTPException(status_code=400, detail="框范围太小，请重新调整")
        conn.execute(
            "UPDATE boxes SET x1=?, y1=?, x2=?, y2=? WHERE id=?",
            (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), box_id),
        )
    decision = payload.decision or box["decision"]
    if decision not in ("PENDING", "ACCEPTED", "REJECTED"):
        raise HTTPException(status_code=400, detail="decision 只能是 PENDING/ACCEPTED/REJECTED")
    type_id = payload.defect_type_id if payload.defect_type_id is not None else box["defect_type_id"]
    if payload.defect_type_id is not None:
        exists = conn.execute("SELECT id FROM defect_types WHERE id=?", (payload.defect_type_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=400, detail="缺陷类型不存在")
    if decision == "ACCEPTED" and not type_id:
        raise HTTPException(status_code=400, detail="接受该框前必须指定缺陷类型")
    annotator = require_annotator(conn, payload.annotator_id) if payload.annotator_id else None
    if decision != "PENDING" and annotator is None and not box["reviewed_by"]:
        raise HTTPException(status_code=400, detail="请先选择人工标注员，再确认该框")
    image_id = box["image_id"]
    prev_status = conn.execute("SELECT status FROM images WHERE id=?", (image_id,)).fetchone()["status"]
    if decision == "PENDING":
        conn.execute(
            "UPDATE boxes SET decision=?, defect_type_id=?, reviewed_by=NULL, reviewed_at=NULL WHERE id=?",
            (decision, type_id, box_id),
        )
        reviewer_id = None
    else:
        reviewer_id = annotator["id"] if annotator else box["reviewed_by"]
        conn.execute(
            "UPDATE boxes SET decision=?, defect_type_id=?, reviewed_by=?, reviewed_at=? WHERE id=?",
            (decision, type_id, reviewer_id, now_text(), box_id),
        )
    status = refresh_image_status(conn, image_id)
    if status == "DONE" and prev_status != "DONE" and reviewer_id:
        conn.execute("UPDATE images SET done_by=?, done_at=? WHERE id=?",
                     (reviewer_id, now_text(), image_id))
    batch_id = conn.execute("SELECT batch_id FROM images WHERE id=?", (image_id,)).fetchone()["batch_id"]
    refresh_batch_status(conn, batch_id)
    return image_detail(conn, image_id)


@app.patch("/api/boxes/{box_id}")
def patch_box(box_id: int, payload: BoxPatch) -> dict[str, Any]:
    with db() as conn:
        return apply_box_patch(conn, box_id, payload)


class BulkAccept(BaseModel):
    only_pending: bool = True
    annotator_id: int | None = None


@app.post("/api/images/{image_id}/accept-suggested")
def accept_suggested(image_id: int, payload: BulkAccept | None = None) -> dict[str, Any]:
    requested = payload or BulkAccept()
    with db() as conn:
        annotator = require_annotator(conn, requested.annotator_id)
        rows = conn.execute(
            "SELECT * FROM boxes WHERE image_id=? AND decision='PENDING'", (image_id,)
        ).fetchall()
        if not rows:
            raise HTTPException(status_code=400, detail="没有待确认的框")
        missing = [row["id"] for row in rows if not row["defect_type_id"]]
        if missing:
            raise HTTPException(status_code=400,
                                detail=f"有 {len(missing)} 个框没有建议类型，请先手动选择缺陷类型")
        prev_status = conn.execute("SELECT status FROM images WHERE id=?", (image_id,)).fetchone()["status"]
        conn.executemany(
            "UPDATE boxes SET decision='ACCEPTED', reviewed_by=?, reviewed_at=? WHERE id=?",
            [(annotator["id"], now_text(), row["id"]) for row in rows],
        )
        status = refresh_image_status(conn, image_id)
        if status == "DONE" and prev_status != "DONE":
            conn.execute("UPDATE images SET done_by=?, done_at=? WHERE id=?",
                         (annotator["id"], now_text(), image_id))
        batch_id = conn.execute("SELECT batch_id FROM images WHERE id=?", (image_id,)).fetchone()["batch_id"]
        refresh_batch_status(conn, batch_id)
        return image_detail(conn, image_id)


class DoneRequest(BaseModel):
    annotator_id: int | None = None


@app.post("/api/images/{image_id}/done")
def mark_image_done(image_id: int, payload: DoneRequest | None = None) -> dict[str, Any]:
    requested = payload or DoneRequest()
    with db() as conn:
        annotator = require_annotator(conn, requested.annotator_id)
        detail = image_detail(conn, image_id)
        if detail["pending"] > 0:
            raise HTTPException(status_code=400, detail=f"还有 {detail['pending']} 个框未确认")
        conn.execute("UPDATE images SET status='DONE', done_by=?, done_at=? WHERE id=?",
                     (annotator["id"], now_text(), image_id))
        refresh_batch_status(conn, detail["batch_id"])
        return image_detail(conn, image_id)


# ---------------------------------------------------------------- AI 标注
def load_model_cached(model_row: sqlite3.Row) -> tuple[Any, dict[int, str], str]:
    model_id = model_row["id"]
    with CACHE_LOCK:
        cached = MODEL_CACHE.get(model_id)
    if cached:
        return cached

    path = Path(model_row["path"])
    if not path.exists():
        raise RuntimeError(f"权重文件不存在：{path}")
    model = build_model_loader(path)
    names = {int(key): str(value) for key, value in (getattr(model, "names", {}) or {}).items()}
    device = device_info()["device"]
    resolved = (model, names, device)
    with CACHE_LOCK:
        MODEL_CACHE[model_id] = resolved
    return resolved


def load_bgr(path: Path):
    import numpy as np

    with Image.open(path) as image:
        array = np.array(image.convert("RGB"))
    return array[:, :, ::-1].copy()


def suggest_type_id(ai_label: str | None, types: list[sqlite3.Row]) -> int | None:
    if not ai_label:
        return None
    key = ai_label.strip().lower().replace(" ", "_").replace("-", "_")
    for item in types:
        code = item["code"].strip().lower().replace(" ", "_").replace("-", "_")
        name = item["name"].strip().lower()
        if key == code or key == name or ai_label.strip() == item["name"]:
            return item["id"]
    for item in types:
        if key in item["name"].strip().lower() or item["name"].strip().lower() in key:
            return item["id"]
    return None


def detect_worker(batch_id: int) -> None:
    job = RUNNING[batch_id]
    try:
        with db() as conn:
            batch = conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
            model_row = conn.execute("SELECT * FROM models WHERE id=?", (batch["model_id"],)).fetchone()
            images = conn.execute("SELECT * FROM images WHERE batch_id=? ORDER BY id", (batch_id,)).fetchall()
            types = conn.execute("SELECT * FROM defect_types WHERE active=1").fetchall()
        model, names, device = load_model_cached(model_row)
        prompt = (batch["prompt"] or "").strip()
        if prompt and is_open_vocab_path(str(model_row["path"])):
            classes = [item.strip() for item in prompt.split(",") if item.strip()]
            if classes:
                model.set_classes(classes)
                names = {int(key): str(value) for key, value in (getattr(model, "names", {}) or {}).items()}
                print(f"[detect] 开放词汇提示词：{classes}", flush=True)
        box_only = (batch["mode"] or "BOX_ONLY").upper() == "BOX_ONLY"
        conf = float(batch["conf"])
        iou = float(batch["iou"])
        for index, image in enumerate(images):
            job["current"] = image["filename"]
            job["processed"] = index
            path = Path(image["path"])
            if not path.exists():
                with db() as conn:
                    conn.execute("UPDATE images SET status='ERROR', error='图片文件丢失' WHERE id=?", (image["id"],))
                continue
            try:
                result = model.predict(source=load_bgr(path), conf=conf, iou=iou,
                                       verbose=False, device=device)[0]
                boxes = getattr(result, "boxes", None)
                rows: list[tuple[Any, ...]] = []
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)
                    for item in range(len(xyxy)):
                        x1, y1, x2, y2 = (float(value) for value in xyxy[item][:4])
                        label = names.get(int(classes[item]), f"class-{int(classes[item])}")
                        rows.append((image["id"], x1, y1, x2, y2, round(float(confs[item]), 5),
                                     label, None if box_only else suggest_type_id(label, types), now_text()))
                with db() as conn:
                    conn.execute("DELETE FROM boxes WHERE image_id=?", (image["id"],))
                    if rows:
                        conn.executemany(
                            "INSERT INTO boxes(image_id,x1,y1,x2,y2,score,ai_label,defect_type_id,decision,created_at) "
                            "VALUES(?,?,?,?,?,?,?,?,'PENDING',?)",
                            rows,
                        )
                    conn.execute("UPDATE images SET status='DETECTED', error=NULL WHERE id=?", (image["id"],))
                    conn.execute("UPDATE batches SET processed=? WHERE id=?", (index + 1, batch_id))
            except Exception as exc:  # noqa: BLE001
                print(f"[detect] 图片 {image['filename']} 失败：{exc}", flush=True)
                with db() as conn:
                    conn.execute("DELETE FROM boxes WHERE image_id=?", (image["id"],))
                    conn.execute("UPDATE images SET status='ERROR', error=? WHERE id=?",
                                 (str(exc)[:500], image["id"]))
            job["processed"] = index + 1
        with db() as conn:
            conn.execute("UPDATE batches SET status='DETECTED', finished_at=?, error=NULL WHERE id=?",
                         (now_text(), batch_id))
            refresh_batch_status(conn, batch_id)
        job["status"] = "DONE"
        job["current"] = ""
        print(f"[detect] 批次 {batch_id} 标注完成，共 {len(images)} 张", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[detect] 批次 {batch_id} 标注失败：{exc}", flush=True)
        job["status"] = "FAILED"
        job["error"] = str(exc)
        with db() as conn:
            conn.execute("UPDATE batches SET status='FAILED', error=?, finished_at=? WHERE id=?",
                         (str(exc)[:500], now_text(), batch_id))


class DetectRequest(BaseModel):
    force: bool = False


@app.post("/api/batches/{batch_id}/detect")
def start_detect(batch_id: int, payload: DetectRequest | None = None) -> dict[str, Any]:
    force = bool(payload.force) if payload else False
    with db() as conn:
        batch = conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
        if not batch:
            raise HTTPException(status_code=404, detail="批次不存在")
        if batch["status"] == "DETECTING":
            raise HTTPException(status_code=409, detail="该批次正在标注中")
        images = conn.execute("SELECT COUNT(*) AS c FROM images WHERE batch_id=?", (batch_id,)).fetchall()
        total = images[0]["c"] if images else 0
        if total == 0:
            raise HTTPException(status_code=400, detail="请先上传图片")
        decided = conn.execute(
            "SELECT COUNT(*) AS c FROM boxes b JOIN images i ON i.id=b.image_id "
            "WHERE i.batch_id=? AND b.decision<>'PENDING'", (batch_id,)
        ).fetchone()["c"]
        if decided > 0 and not force:
            raise HTTPException(status_code=409,
                                detail=f"已有 {decided} 个框被人工确认，重新标注会清空全部结果；如确认请选择覆盖")
        running = [key for key, value in RUNNING.items()
                   if key != "probe" and isinstance(value, dict) and value.get("status") == "RUNNING"]
        if running:
            raise HTTPException(status_code=409, detail="已有批次正在标注，请等待完成后再开始")
        conn.execute(
            "UPDATE batches SET status='DETECTING', total=?, processed=0, error=NULL, "
            "started_at=?, finished_at=NULL WHERE id=?",
            (total, now_text(), batch_id),
        )
    RUNNING[batch_id] = {"status": "RUNNING", "total": total, "processed": 0,
                         "current": "", "error": None, "started_at": now_text()}
    thread = threading.Thread(target=detect_worker, args=(batch_id,), name=f"detect-{batch_id}", daemon=True)
    thread.start()
    return {"ok": True, "total": total}


@app.get("/api/batches/{batch_id}/progress")
def batch_progress(batch_id: int) -> dict[str, Any]:
    with db() as conn:
        batch = conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
    if not batch:
        raise HTTPException(status_code=404, detail="批次不存在")
    job = RUNNING.get(batch_id)
    if job and job.get("status") in ("RUNNING", "FAILED"):
        return {"status": job["status"], "total": job["total"], "processed": job["processed"],
                "current": job.get("current"), "error": job.get("error")}
    return {"status": batch["status"], "total": batch["total"], "processed": batch["processed"],
            "current": "", "error": batch["error"]}


# ---------------------------------------------------------------- 统计与导出
def compute_stats(batch_id: int) -> dict[str, Any]:
    with db() as conn:
        batch = conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
        if not batch:
            raise HTTPException(status_code=404, detail="批次不存在")
        model = conn.execute("SELECT * FROM models WHERE id=?", (batch["model_id"],)).fetchone()
        rows = conn.execute(
            """
            SELECT i.id, i.filename, i.status, i.width, i.height,
                   COUNT(b.id) AS total,
                   COALESCE(SUM(CASE WHEN b.decision='ACCEPTED' THEN 1 ELSE 0 END),0) AS accepted,
                   COALESCE(SUM(CASE WHEN b.decision='REJECTED' THEN 1 ELSE 0 END),0) AS rejected,
                   COALESCE(SUM(CASE WHEN b.decision='PENDING' THEN 1 ELSE 0 END),0) AS pending
            FROM images i LEFT JOIN boxes b ON b.image_id=i.id
            WHERE i.batch_id=? GROUP BY i.id ORDER BY i.id
            """,
            (batch_id,),
        ).fetchall()
        by_type = conn.execute(
            """
            SELECT t.id, t.code, t.name, t.color, COUNT(b.id) AS count
            FROM boxes b JOIN images i ON i.id=b.image_id
            JOIN defect_types t ON t.id=b.defect_type_id
            WHERE i.batch_id=? AND b.decision='ACCEPTED'
            GROUP BY t.id ORDER BY count DESC
            """,
            (batch_id,),
        ).fetchall()
        avg_score = conn.execute(
            "SELECT AVG(b.score) AS v FROM boxes b JOIN images i ON i.id=b.image_id WHERE i.batch_id=?",
            (batch_id,),
        ).fetchone()["v"]
        exports = conn.execute(
            "SELECT * FROM exports WHERE batch_id=? ORDER BY id DESC", (batch_id,)
        ).fetchall()
        by_annotator = conn.execute(
            """
            SELECT a.id, a.name, a.code, a.active,
              COALESCE(SUM(CASE WHEN b.decision='ACCEPTED' THEN 1 ELSE 0 END),0) AS accepted,
              COALESCE(SUM(CASE WHEN b.decision='REJECTED' THEN 1 ELSE 0 END),0) AS rejected,
              (SELECT COUNT(*) FROM images i2 WHERE i2.batch_id=? AND i2.done_by=a.id) AS done_images
            FROM annotators a
            LEFT JOIN boxes b ON b.reviewed_by=a.id
              AND b.image_id IN (SELECT id FROM images WHERE batch_id=?)
            GROUP BY a.id ORDER BY accepted DESC, a.id
            """,
            (batch_id, batch_id),
        ).fetchall()
        unassigned = conn.execute(
            "SELECT COUNT(*) AS c FROM boxes b JOIN images i ON i.id=b.image_id "
            "WHERE i.batch_id=? AND b.reviewed_by IS NULL AND b.decision<>'PENDING'",
            (batch_id,),
        ).fetchone()["c"]
    images = [row_to_dict(row) for row in rows]
    total_boxes = sum(item["total"] for item in images)
    accepted = sum(item["accepted"] for item in images)
    rejected = sum(item["rejected"] for item in images)
    pending = sum(item["pending"] for item in images)
    return {
        "batch_id": batch_id,
        "name": batch["name"],
        "status": batch["status"],
        "model": model_to_dict(model),
        "created_at": batch["created_at"],
        "started_at": batch["started_at"],
        "finished_at": batch["finished_at"],
        "images": len(images),
        "images_done": sum(1 for item in images if item["status"] == "DONE"),
        "images_empty": sum(1 for item in images if item["total"] == 0),
        "boxes": total_boxes,
        "accepted": accepted,
        "rejected": rejected,
        "negatives": rejected,
        "pending": pending,
        "accept_rate": round(accepted / total_boxes, 4) if total_boxes else 0,
        "avg_score": round(avg_score, 4) if avg_score is not None else None,
        "by_type": [row_to_dict(row) for row in by_type],
        "by_annotator": [row_to_dict(row) for row in by_annotator],
        "unassigned_boxes": unassigned,
        "exports": [row_to_dict(row) for row in exports],
        "image_rows": images,
        "device": device_info(),
    }


@app.get("/api/batches/{batch_id}/stats")
def batch_stats(batch_id: int) -> dict[str, Any]:
    stats = compute_stats(batch_id)
    stats.pop("image_rows", None)
    return stats


def load_font(size: int):
    for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                      "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def render_annotated(source: Path, target: Path, boxes: list[dict[str, Any]],
                     width: int, height: int, colors: dict[int, str]) -> None:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, int(min(width, height) / 350))
    font = load_font(max(14, int(min(width, height) / 42)))
    for box in boxes:
        color = colors.get(box.get("defect_type_id") or -1, "#4f8cff")
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        label = box.get("defect_name") or box.get("ai_label") or "defect"
        if box.get("score") is not None:
            label = f"{label} {box['score']:.2f}"
        try:
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = right - left, bottom - top
        except Exception:  # noqa: BLE001
            text_w, text_h = len(label) * 8, 16
        text_y = y1 - text_h - 8 if y1 - text_h - 8 > 0 else y1 + 2
        draw.rectangle([x1, text_y, x1 + text_w + 10, text_y + text_h + 6], fill=color)
        draw.text((x1 + 5, text_y + 3), label, fill="#ffffff", font=font)
    image.save(target, quality=92)


def yolo_line(box: dict[str, Any], class_index: int, width: int, height: int) -> str:
    x1 = max(0.0, min(float(box["x1"]), width))
    x2 = max(0.0, min(float(box["x2"]), width))
    y1 = max(0.0, min(float(box["y1"]), height))
    y2 = max(0.0, min(float(box["y2"]), height))
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    bw = max(0.0, (x2 - x1) / width)
    bh = max(0.0, (y2 - y1) / height)
    return f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def csv_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    if any(char in text for char in ',"\n'):
        text = '"' + text.replace('"', '""') + '"'
    return text


@app.post("/api/batches/{batch_id}/export")
def export_batch(batch_id: int) -> dict[str, Any]:
    stats = compute_stats(batch_id)
    with db() as conn:
        types = conn.execute("SELECT * FROM defect_types ORDER BY id").fetchall()
        images = conn.execute("SELECT * FROM images WHERE batch_id=? ORDER BY id", (batch_id,)).fetchall()
        box_rows = conn.execute(
            "SELECT b.*, t.code AS type_code, t.name AS type_name, t.color AS type_color, i.filename, i.width, i.height "
            "FROM boxes b JOIN images i ON i.id=b.image_id "
            "LEFT JOIN defect_types t ON t.id=b.defect_type_id WHERE i.batch_id=? ORDER BY b.id",
            (batch_id,),
        ).fetchall()
        batch = conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
        model = conn.execute("SELECT * FROM models WHERE id=?", (batch["model_id"],)).fetchone()
        annotators = {item["id"]: dict(item) for item in
                      conn.execute("SELECT id,name,code FROM annotators").fetchall()}
    if batch["status"] == "DETECTING":
        raise HTTPException(status_code=409, detail="批次正在 AI 标注中，请等待完成后再导出")
    boxes_by_image: dict[int, list[dict[str, Any]]] = {}
    accepted_types: dict[int, sqlite3.Row] = {}
    for row in box_rows:
        item = row_to_dict(row)
        boxes_by_image.setdefault(item["image_id"], []).append(item)
        if item["decision"] == "ACCEPTED" and item["defect_type_id"]:
            accepted_types[item["defect_type_id"]] = row
    class_list = [{"index": index, "id": type_id, "code": accepted_types[type_id]["type_code"],
                   "name": accepted_types[type_id]["type_name"], "color": accepted_types[type_id]["type_color"]}
                  for index, type_id in enumerate(sorted(accepted_types))]
    class_index = {item["id"]: item["index"] for item in class_list}
    colors = {item["id"]: item["color"] for item in class_list}
    type_names = {item["id"]: item["name"] for item in class_list}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"batch{batch_id}_{stamp}.zip"
    zip_path = EXPORT_DIR / zip_name
    annotations_images = []
    csv_lines = ["filename,width,height,boxes,accepted,rejected,pending,defect_types"]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for image in images:
            source = Path(image["path"])
            if not source.exists():
                continue
            items = boxes_by_image.get(image["id"], [])
            accepted = [item for item in items if item["decision"] == "ACCEPTED"]
            exported = []
            for item in accepted:
                item["defect_name"] = type_names.get(item["defect_type_id"])
                exported.append(item)
            annotate_target = EXPORT_DIR / f"tmp_{image['id']}.jpg"
            try:
                render_annotated(source, annotate_target, exported, image["width"], image["height"], colors)
                archive.write(annotate_target, f"annotated/{image['id']}_{Path(image['filename']).stem}.jpg")
            finally:
                annotate_target.unlink(missing_ok=True)
            lines = [yolo_line(item, class_index[item["defect_type_id"]], image["width"], image["height"])
                     for item in accepted if item["defect_type_id"] in class_index]
            archive.writestr(f"labels/{image['id']}_{Path(image['filename']).stem}.txt",
                             "\n".join(lines) + ("\n" if lines else ""))
            types_text = "|".join(sorted({item["defect_name"] for item in exported if item.get("defect_name")}))
            csv_lines.append(
                f"{image['filename']},{image['width']},{image['height']},{len(items)},"
                f"{len(accepted)},{sum(1 for item in items if item['decision']=='REJECTED')},"
                f"{sum(1 for item in items if item['decision']=='PENDING')},{types_text}"
            )
            annotations_images.append({
                "id": image["id"], "filename": image["filename"], "status": image["status"],
                "width": image["width"], "height": image["height"],
                "done_by": image["done_by"],
                "done_by_name": (annotators.get(image["done_by"]) or {}).get("name"),
                "done_at": image["done_at"],
                "boxes": [{
                    "id": item["id"], "x1": round(item["x1"], 2), "y1": round(item["y1"], 2),
                    "x2": round(item["x2"], 2), "y2": round(item["y2"], 2),
                    "score": item["score"], "ai_label": item["ai_label"], "decision": item["decision"],
                    "reviewed_by": item["reviewed_by"],
                    "reviewer": annotators.get(item["reviewed_by"]),
                    "reviewed_at": item["reviewed_at"],
                    "defect_type": None if not item["defect_type_id"] else {
                        "id": item["defect_type_id"], "code": item["type_code"],
                        "name": item["type_name"], "color": item["type_color"]},
                } for item in items],
            })
        archive.writestr("classes.txt", "\n".join(
            f"{item['index']} {item['name']} ({item['code']})" for item in class_list) + "\n")
        archive.writestr("annotations.json", json.dumps({
            "batch": {"id": batch["id"], "name": batch["name"], "status": batch["status"],
                      "mode": batch["mode"], "prompt": batch["prompt"],
                      "created_at": batch["created_at"], "started_at": batch["started_at"],
                      "finished_at": batch["finished_at"]},
            "model": {"id": model["id"], "name": model["name"], "task": model["task"],
                      "conf": batch["conf"], "iou": batch["iou"]},
            "classes": class_list,
            "annotators": stats["by_annotator"],
            "images": annotations_images,
            "stats": {key: value for key, value in stats.items()
                      if key not in ("image_rows", "exports", "device")},
            "generated_at": now_text(),
        }, ensure_ascii=False, indent=2))
        archive.writestr("stats.csv", "\n".join(csv_lines) + "\n")
        annotator_lines = ["name,code,accepted_boxes,rejected_boxes,done_images"]
        for item in stats["by_annotator"]:
            annotator_lines.append(",".join(csv_cell(value) for value in (
                item["name"], item["code"], item["accepted"], item["rejected"], item["done_images"])))
        archive.writestr("annotators.csv", "\n".join(annotator_lines) + "\n")
        archive.writestr("stats.json", json.dumps(
            {key: value for key, value in stats.items() if key not in ("image_rows", "exports", "device")},
            ensure_ascii=False, indent=2))
    with db() as conn:
        conn.execute(
            "INSERT INTO exports(batch_id,filename,path,created_at) VALUES(?,?,?,?)",
            (batch_id, zip_name, str(zip_path), now_text()),
        )
    return {"filename": zip_name, "url": f"/api/exports/{zip_name}", "stats": {
        key: value for key, value in stats.items() if key not in ("image_rows", "exports", "device")}}


@app.get("/api/exports/{filename}")
def download_export(filename: str):
    path = (EXPORT_DIR / filename).resolve()
    if not path.is_relative_to(EXPORT_DIR.resolve()) or not path.exists():
        raise HTTPException(status_code=404, detail="导出文件不存在")
    return FileResponse(path, filename=path.name, media_type="application/zip")


# ---------------------------------------------------------------- 静态页面
@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)
