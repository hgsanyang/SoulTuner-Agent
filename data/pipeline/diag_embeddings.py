"""Cross-check: audio files on disk vs Song nodes in Neo4j"""
import sys, os, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from retrieval.neo4j_client import get_neo4j_client

AUDIO_DIR = r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data\processed_audio\audio"

# 1. Count audio files on disk
audio_files = []
for ext in ("*.mp3", "*.flac", "*.wav", "*.ogg"):
    audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, ext)))
audio_files = list(set(audio_files))
print(f"Audio files on disk: {len(audio_files)}")

# 2. Count Song nodes in Neo4j
client = get_neo4j_client()
total = client.execute_query("MATCH (s:Song) RETURN count(s) AS c")
print(f"Song nodes in Neo4j: {total[0]['c']}")

# 3. Get all music_ids from Neo4j
neo4j_titles = set()
records = client.execute_query("MATCH (s:Song) RETURN s.title AS t")
for r in records:
    neo4j_titles.add(r['t'])

# 4. Find audio files NOT in Neo4j (by title matching)
missing_in_neo4j = []
for af in sorted(audio_files):
    basename = os.path.splitext(os.path.basename(af))[0]
    # title is usually before " - " in the basename
    title = basename.split(" - ")[0].strip() if " - " in basename else basename
    if title not in neo4j_titles:
        missing_in_neo4j.append(basename)

print(f"\nAudio files NOT matching any Neo4j title: {len(missing_in_neo4j)}")
if missing_in_neo4j[:15]:
    print("Sample (first 15):")
    for m in missing_in_neo4j[:15]:
        print(f"  {m}")
