# 缺陷标注系统（AI-Labeling-system）

独立的缺陷标注服务：FastAPI + SQLite + 原生前端，单机/GPU 服务器运行，
可通过 Cloudflare Tunnel 暴露成固定域名的“点开即用”网站。

数据全部保存在本机（SQLite + 本地文件），不需要 Docker、CVAT 或 PostgreSQL。

## 功能

1. 批量导入待标注图片（JPG / PNG / BMP / WEBP / TIFF）
2. 选择 AI 模型：本机 7 类缺陷模型、YOLO-World、YOLOE、Grounding DINO、OWLv2
3. 配置缺陷类型与颜色
4. AI 只框出候选区域，人工逐框确认类型，可标记「不是缺陷」
5. 人工可拖动/缩放候选框，也可新增框
6. 记录标注员与操作时间，按标注员统计
7. 导出 ZIP：画框图、YOLO 标签、`annotations.json`、`classes.txt`、统计 CSV/JSON

## 环境要求

- Python 3.10+
- 可选 NVIDIA GPU（CUDA 版 PyTorch）；CPU 也能跑，但大模型较慢
- 磁盘：仅 7 类模型约 50 MB；Grounding DINO / OWLv2 各约 600 MB

## 安装

```powershell
git clone https://github.com/jxz001110-del/AI-Labeling-system.git
cd AI-Labeling-system
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果使用 GPU，先按显卡 CUDA 版本安装 PyTorch，例如：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

也可以直接复用系统里已有的虚拟环境，只要装齐 `requirements.txt`。

## 模型放置

服务启动时会自动登记以下权重，文件不存在则跳过（模型文件不进 Git 仓库）。

| 模型 | 放置位置 |
|---|---|
| 可见光缺陷分割（7 类） | `models/visible_light.pt` |
| 中文缺陷检测（7 类） | `models/Object Detection.pt` |
| 裂缝检测（单类） | `models/best.pt` |
| YOLO-World 开放词汇 | `models/yolov8s-worldv2.pt` |
| YOLOE-26S / 26M / 26L | `models/yoloe-26{s,m,l}-seg.pt` |
| Grounding DINO Tiny | `models/grounding-dino-tiny/`（Hugging Face 快照目录） |
| OWLv2 Base Ensemble | `models/owlv2-base-patch16-ensemble/`（Hugging Face 快照目录） |
| YOLOE 文本编码器 | 仓库根目录 `mobileclip2_b.ts`（约 242 MB） |

### 新机器：一键下载开放词汇模型

在仓库根目录执行：

```powershell
python scripts/download_models.py
```

脚本会：

- 从 Ultralytics 官方资源下载 YOLO-World、YOLOE-26S/M/L 和 `mobileclip2_b.ts`；
- 从 Hugging Face 下载 Grounding DINO Tiny、OWLv2 Base Ensemble；
- 已存在的文件自动跳过，单个文件失败不影响其他文件。

国内网络建议先设置镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python scripts/download_models.py
```

### 本机 7 类权重

`visible_light.pt`、`Object Detection.pt`、`best.pt` 是自训练权重，不在网上公开，
需要从旧机器拷贝到 `models/`，或在「模型库 → 上传 .pt 权重」中上传。
脚本只会提示缺失，不会伪造文件。

### 手动下载

| 文件 | 来源 | 放到 |
|---|---|---|
| `yolov8s-worldv2.pt` | Ultralytics 官方资源 | `models/` |
| `yoloe-26s-seg.pt` / `26m` / `26l` | Ultralytics 官方资源 | `models/` |
| `mobileclip2_b.ts` | Ultralytics 官方资源 | 仓库根目录 |
| Grounding DINO Tiny | `IDEA-Research/grounding-dino-tiny` | `models/grounding-dino-tiny/` |
| OWLv2 Base Ensemble | `google/owlv2-base-patch16-ensemble` | `models/owlv2-base-patch16-ensemble/` |

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python -c "from huggingface_hub import snapshot_download; snapshot_download('IDEA-Research/grounding-dino-tiny', local_dir='models/grounding-dino-tiny')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/owlv2-base-patch16-ensemble', local_dir='models/owlv2-base-patch16-ensemble')"
```

### 离线机器

在旧机器完整拷贝 `models/` 目录和 `mobileclip2_b.ts` 到新机器相同位置，
再启动服务即可。全部权重约 1.7GB。

## 本地启动

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8020
```

- 打开：http://127.0.0.1:8020
- 健康检查：http://127.0.0.1:8020/api/health
- 日志目录：`data/`

## 通过 Cloudflare Tunnel 变成固定域名

前置条件：一个托管在 Cloudflare 的域名。

```powershell
# 1. 安装 cloudflared：winget install --id Cloudflare.cloudflared
# 2. 浏览器登录（会打开网页选 Zone）
cloudflared tunnel login

# 3. 创建命名隧道
cloudflared tunnel create labeler

# 4. 绑定域名（示例：labeler.example.com 指向本机 8020）
cloudflared tunnel route dns labeler labeler.example.com

# 5. 运行（配置见下方 config.yml）
cloudflared tunnel run labeler
```

`%USERPROFILE%\.cloudflared\config.yml`：

```yaml
tunnel: labeler
credentials-file: C:\Users\<用户名>\.cloudflared\<tunnel-id>.json

ingress:
  - hostname: labeler.example.com
    service: http://127.0.0.1:8020
  - service: http_status:404
```

Windows 常驻（开机自启）：

```powershell
cloudflared service install
```

> 公开到公网前请先加访问控制（Cloudflare Access 或应用内登录），
> 否则任何拿到域名的人都能标注或删数据。

## 数据与导出

| 路径 | 内容 |
|---|---|
| `data/labeler.db` | SQLite：批次、图片、框、缺陷类型、标注员 |
| `data/uploads/` | 原始上传图片 |
| `exports/` | 导出的 ZIP 包 |

导出 ZIP 包含：

- `annotated/`：画好框的图片
- `labels/`：YOLO 格式 txt
- `annotations.json`：逐框类型、决策、标注员、来源（AI/人工）
- `classes.txt`
- `stats.csv` / `stats.json`

只有人工接受了类型的框会进入 YOLO 标签；标记为「不是缺陷」的框保留在
`annotations.json` 中，不进入训练标签。

## 注意事项

- 本服务默认没有登录鉴权。只在本机调试时监听 `127.0.0.1`；
  通过 Cloudflare Tunnel 暴露到公网时，务必额外加访问控制。
- 首次加载 Grounding DINO / OWLv2 需要几秒，之后单图约 0.2 秒（RTX 5060 实测）。
- 开放词汇模型分数偏低，OWLv2 建议置信度 0.04–0.08；Grounding DINO 建议 0.10–0.20。
- 本服务不依赖 Docker 和 CVAT。
