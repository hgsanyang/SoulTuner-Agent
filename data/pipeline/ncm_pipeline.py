import os
import sys
import struct
import binascii
import json
import base64
import shutil
import concurrent.futures
from pathlib import Path
from Crypto.Cipher import AES
import mutagen
from mutagen.id3 import ID3, APIC, TIT2, TPE1, TALB
from mutagen.flac import FLAC, Picture
from mutagen.mp3 import MP3

def build_key_box(key):
    """
    用 NCM 解密出来的密钥数据构建 RC4 Key Box
    """
    box = bytearray(range(256))
    j = 0
    key_len = len(key)
    for i in range(256):
        j = (box[i] + j + key[i % key_len]) & 0xff
        box[i], box[j] = box[j], box[i]
    return box

def decrypt_ncm(file_path: str, output_dir: str):
    """
    解码 NCM 文件，提取封面、音频、元数据，并转换为标准格式。
    返回: 转换后的文件路径, 提取的元信息 JSON 字典
    """
    core_key = binascii.a2b_hex("687a4852416d736f356b496e62617857")
    meta_key = binascii.a2b_hex("2331346C6A6B5F215C5D2630553C2728")
    unpad = lambda s: s[0:-(s[-1] if type(s[-1]) == int else ord(s[-1]))]
    
    with open(file_path, 'rb') as f:
        # 1. 验证 Magic Header
        header = f.read(8)
        if binascii.b2a_hex(header) != b'4354454e4644414d':
            raise ValueError(f"文件不是合法的 NCM 文件")
            
        f.seek(2, 1) # Skip gap
        
        # 2. 解密 RC4 密钥
        key_length = struct.unpack('<I', f.read(4))[0]
        key_data = bytearray(f.read(key_length))
        for i in range(len(key_data)): key_data[i] ^= 0x64
        cryptor = AES.new(core_key, AES.MODE_ECB)
        key_data = unpad(cryptor.decrypt(bytes(key_data)))[17:]
        key_box = build_key_box(key_data)
        
        # 3. 解密 metadata 
        meta_length = struct.unpack('<I', f.read(4))[0]
        meta_data_dict = {}
        if meta_length:
            meta_data = bytearray(f.read(meta_length))
            for i in range(len(meta_data)): meta_data[i] ^= 0x63
            meta_data = base64.b64decode(bytes(meta_data[22:]))
            cryptor = AES.new(meta_key, AES.MODE_ECB)
            meta_data_text = unpad(cryptor.decrypt(meta_data)).decode('utf-8')[6:]
            meta_data_dict = json.loads(meta_data_text)
            
        # 4. 提取封面图片数据
        crc32 = struct.unpack('<I', f.read(4))[0]
        f.seek(5, 1)
        image_size = struct.unpack('<I', f.read(4))[0]
        image_data = f.read(image_size)
        
        # 5. 生成解密后的音频文件
        song_title = meta_data_dict.get("musicName", "Unknown")
        artist_names = "，".join([t[0] for t in meta_data_dict.get("artist", [["Unknown"]])])
        ext = meta_data_dict.get("format", "mp3")
        
        safe_title = "".join(c for c in song_title if c not in r'\/:*?"<>|')
        safe_artist = "".join(c for c in artist_names if c not in r'\/:*?"<>|')
        file_basename = f"{safe_title} - {safe_artist}"
        file_name = f"{file_basename}.{ext}"
        
        # 构建分类存放的子目录
        audio_dir = os.path.join(output_dir, "audio")
        cover_dir = os.path.join(output_dir, "covers")
        lrc_dir = os.path.join(output_dir, "lyrics")
        meta_dir = os.path.join(output_dir, "metadata")
        
        output_path = os.path.join(audio_dir, file_name)
        cover_path = os.path.join(cover_dir, f"{file_basename}_cover.jpg")
        lrc_output_path = os.path.join(lrc_dir, f"{file_basename}.lrc")
        meta_output_path = os.path.join(meta_dir, f"{file_basename}_meta.json")
        
        # 【防重机制】：如果目标音频文件存在，跳过流解码以节省时间
        if os.path.exists(output_path):
            return output_path, meta_data_dict, True
            
        with open(output_path, 'wb') as fout:
            while True:
                chunk = bytearray(f.read(0x8000))
                if not chunk: break
                for i in range(1, len(chunk) + 1):
                    j = i & 0xff
                    chunk[i - 1] ^= key_box[(key_box[j] + key_box[(key_box[j] + j) & 0xff]) & 0xff]
                fout.write(chunk)
                
    # 6. 把封面图、歌名等写入音频内嵌标签
    write_audio_metadata(output_path, ext, meta_data_dict, image_data)
    
    # 7. 单独保存提取出来的封面大图 (显式导出)
    if image_data:
        try:
            with open(cover_path, 'wb') as f_img:
                f_img.write(image_data)
        except Exception:
            pass

    # 8. 寻找同目录下的同名 .lrc 歌词文件并移动/拷贝到输出目录
    original_lrc_path = file_path.replace('.ncm', '.lrc')
    if os.path.exists(original_lrc_path):
        try:
            shutil.copy2(original_lrc_path, lrc_output_path)
        except Exception:
            pass
    # 9. 导出原生的元数据 JSON (包含真实ID、比特率、完整专辑信息等)
    # 这对于后续录入 Neo4j 图谱 (特别是保留网易云 ID 作为唯一标识) 非常有用
    try:
        with open(meta_output_path, 'w', encoding='utf-8') as f_meta:
            json.dump(meta_data_dict, f_meta, ensure_ascii=False, indent=4)
    except Exception:
        pass
            
    return output_path, meta_data_dict, False

def write_audio_metadata(file_path, ext, meta_data, image_data):
    title = meta_data.get("musicName", "Unknown")
    artist = "，".join([t[0] for t in meta_data.get("artist", [["Unknown"]])])
    album = meta_data.get("album", "Unknown")
    
    try:
        if ext == "mp3":
            audio = MP3(file_path, ID3=ID3)
            if audio.tags is None:
                audio.add_tags()
            if image_data:
                audio.tags.add(APIC(encoding=3, mime='image/jpeg', type=3, desc='Cover', data=image_data))
            audio.tags.add(TIT2(encoding=3, text=title))
            audio.tags.add(TPE1(encoding=3, text=artist))
            audio.tags.add(TALB(encoding=3, text=album))
            audio.save()
        elif ext == "flac":
            audio = FLAC(file_path)
            if image_data:
                pic = Picture()
                pic.type = 3
                pic.desc = "Cover"
                pic.mime = "image/jpeg"
                pic.data = image_data
                audio.add_picture(pic)
            audio["title"] = title
            audio["artist"] = artist
            audio["album"] = album
            audio.save()
    except Exception:
        pass

def process_single_file(ncm_file, output_dir):
    try:
        out_file, meta, skipped = decrypt_ncm(str(ncm_file), output_dir)
        filename = os.path.basename(out_file)
        if skipped:
            return f"⏭️ 已存在 -> {filename}"
        else:
            return f"✅ 成功提取 ({meta.get('format', 'mp3').upper()}) -> {filename}"
    except Exception as e:
        return f"❌ 失败 {os.path.basename(str(ncm_file))}: {e}"

def process_ncm_directory(input_dir: str, output_dir: str):
    # 提前创建好所有的分类目录
    for sub_dir in ["audio", "covers", "lyrics", "metadata"]:
        os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
        
    input_path = Path(input_dir)
    
    # 递归查找所有 .ncm，解决可能存在的子文件夹问题
    ncm_files = list(input_path.rglob("*.ncm"))
    if not ncm_files:
        print(f"在 {input_dir} 及其子目录下没有找到 .ncm 文件")
        return
        
    print(f"找到 {len(ncm_files)} 个 NCM 文件，启动多线程并发处理...")
    
    # 使用 ThreadPoolExecutor 并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_file, ncm_file, output_dir) for ncm_file in ncm_files]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
            
    print("================== 处理完成 ==================")

if __name__ == "__main__":
    # 根目录，rglob 会自动搜索包括 VipSongsDownload 在内的所有子目录
    INPUT_DIR = r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data\raw_ncm"
    OUTPUT_DIR = r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data\processed_audio"
    
    print("================== 网易云 NCM 自动化处理器 (并行版) ==================")
    print(f"正在扫描: {INPUT_DIR} 及其所有子文件夹")
    print(f"输出目录: {OUTPUT_DIR}")
    
    process_ncm_directory(INPUT_DIR, OUTPUT_DIR)
