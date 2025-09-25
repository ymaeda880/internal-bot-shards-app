# internal-bot-app

# 元データの場所

- データをおくフォルダーのパスが Home,Portable,PrecMacmini,PrecServer,other1 によって異なるので，ラジオボタンを sidebar に出して選択．
- フォルダーパスは，config フォルダの中に置き，ファイル名は config.py とする．

- 外部 SSD ファイル（ファイルパス:ssd_path）

  - Home：/Volumes/Extreme\ SSD/bot_data
  - Portable：別のパス+/bot_data
  - PrecMacmini：別のパス+/bot_data
  - PrecServer：別のパス+/bot_data

- 入力：ssd_path/bot_data/pdf/<shard id>（通常は年度フォルダ，例:2025）
-
- 出力：data/vectorstore/<backend>/<shard_id>/
  - vectors.npy（float32）
  - meta.jsonl（jsonl）
  - processed_files.json（すでに取り込んだ PDF の記録）

backend は UI のラジオで選択（例：openai or local）。バックエンドごとに別フォルダに分けるため、切り替えても干渉しません。

- pdf データの場所：bot_data/pdf

- バックアップデータの場所：ssd_path/bot_data/backup
- vectorestore の場所は今まで通り
