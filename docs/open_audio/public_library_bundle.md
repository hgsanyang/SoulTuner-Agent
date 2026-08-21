# Song Describer 公开曲库封面与 ModelScope 打包

## 已盘点的数据

- 本地完整音频：706 首，约 3.111 GiB。
- 上游归档：Zenodo `audio.zip`，MD5 `2126b8facfe9468cf806c6154e09bbe5`，已匹配。
- 完整清单：1106 条 captions，逐曲音频许可证、归属和 SHA-256 均有记录。

## 封面来源与许可边界

Jamendo 官方 tracks/albums API 文档给出的封面地址格式是：

```text
https://usercontent.jamendo.com/?type=album&id=<album_id>&width=600&trackid=<track_id>
```

脚本使用这个固定 600px 端点取得 JPEG，只验证媒体类型、JPEG 尺寸和 SHA-256，不在本地
转码、裁剪或重编码。请求不存在的 album ID 可能返回 Jamendo 默认 JPEG，也可能由边缘节点
返回 GIF 错误资源；脚本会尝试记录默认 JPEG 的 SHA-256，并始终以 MIME + JPEG 尺寸双重
校验兜底。相同默认 SHA、非 JPEG 或无有效尺寸都会判定为“源端无封面”。

重要：Song Describer 的逐曲 CC 许可证证明的是发布包内的音频，不应自动扩展到 Jamendo
专辑美术作品。Jamendo API 条款还要求创作者/Jamendo 归属、逐内容回链，并限制专门的离线
缓存服务。因此：

- `runtime_cache/*.jpg` 只用于可用性审计和合理运行时缓存；
- ModelScope 公共曲库不打包或再分发这些 JPEG；
- `cover_url` 保留官方远程展示 URL；
- `cover_attribution` 和 `cover_source_page_url` 必须同时展示；
- 每首曲目都有 SoulTuner 稳定生成的本地 SVG 回退封面；
- 源端无图、返回默认图或请求失败时，直接使用本地 SVG。

Jamendo API 条款：https://devportal.jamendo.com/api_terms_of_use

## 准备完整 706 首封面审计

```powershell
$cache = "C:\Users\sanyang\sanyangworkspace\music_recommendation\data\mtg_sample"
$manifest = "$cache\artifacts\song_describer_full.jsonl"
python -m tools.data.song_describer_public_bundle covers `
  --audio-manifest $manifest `
  --cover-root "$cache\covers" `
  --workers 8
```

该命令可重复运行：已有运行时 JPEG 会复核后复用；`--refresh` 才强制重新请求。

产物：

- `covers/cover_manifest_full.jsonl`
- `covers/cover_audit_full.json`
- `covers/runtime_cache/*.jpg`（禁止进入公共包）
- `covers/placeholders/*.svg`（可进入公共包，CC-BY-4.0）

## 构建 ModelScope 公共曲库目录

```powershell
python -m tools.data.song_describer_public_bundle bundle `
  --audio-manifest "$cache\artifacts\song_describer_full.jsonl" `
  --cover-manifest "$cache\covers\cover_manifest_full.jsonl" `
  --cache-dir $cache `
  --output-dir "C:\Users\sanyang\sanyangworkspace\music_recommendation\data\modelscope_public_library" `
  --mode hardlink
```

`hardlink` 不复制第二份 3.1 GiB 音频数据；跨盘失败时自动退回普通复制。正式移动到其他磁盘
或上传工具不识别硬链接时可用 `--mode copy`。

公共目录包含：

- `catalog.jsonl`：音频、captions、逐曲许可证、封面远程 URL 与回退路径；
- `audio/`：原始 MP3 字节，不转码、不裁剪；
- `covers/placeholders/`：稳定生成的 SVG；
- `bundle_audit.json`：体积、覆盖率、物化方式与 catalog SHA-256；
- `README.md`：ModelScope 数据集卡片草稿。

本流程只构建本地产物，不执行上传或创空间部署。
