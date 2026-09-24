#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相机推流接收验证工具 (纯 PIL 版，彻底绕开 OpenCV 环境冲突)"""
import argparse, base64, json, os, threading, time, io
from PIL import Image
import zmq

RESULT, LIVE, LOCK = {}, {}, threading.Lock()

def recv_one(host, port, duration):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 3000)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(f"tcp://{host}:{port}")
    n, keys, shape, nbytes = 0, set(), None, 0
    t0 = time.time()
    while time.time() - t0 < duration:
        try:
            msg = sock.recv_string()
        except zmq.Again:
            break
        try:
            data = json.loads(msg)
        except Exception:
            continue
        for key, b64 in data.get("images", {}).items():
            raw = base64.b64decode(b64)
            nbytes += len(raw)
            try:
                # 使用 PIL 解码 JPEG 字节流
                img = Image.open(io.BytesIO(raw))
                img.load()  # 强制完成解码
                n += 1
                keys.add(key)
                shape = (img.size[1], img.size[0], 3)  # H, W, C
                with LOCK:
                    LIVE[key] = img
            except Exception:
                pass
    dt = time.time() - t0
    with LOCK:
        RESULT[port] = {"frames": n, "keys": keys, "shape": shape,
                        "fps": (n / dt) if dt > 0 else 0.0, "mb": nbytes / 1e6}
    sock.close()
    ctx.term()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ports", default="5556,5557,5558")
    ap.add_argument("--duration", type=float, default=5.0)
    args = ap.parse_args()

    ports = [int(p) for p in args.ports.split(",") if p.strip()]
    print(f"[*] 连接 tcp://{args.host} 端口 {ports}, 采样 {args.duration}s ...")
    
    threads = [threading.Thread(target=recv_one,
                                args=(args.host, p, args.duration), daemon=True)
               for p in ports]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=args.duration + 2)

    print("\n" + "=" * 66)
    ok = True
    for p in ports:
        r = RESULT.get(p)
        if not r or r["frames"] == 0:
            print(f"[x] 端口 {p}: 未收到任何画面")
            ok = False
            continue
        shp = r["shape"]
        print(f"[+] 端口 {p}: 收到 {r['frames']} 帧 | 键名: {','.join(sorted(r['keys']))} | "
              f"尺寸: {shp[1]}x{shp[0]} | ~{r['fps']:.1f} fps")

    # 自动保存快照到本地文件夹
    save_dir = "./stream_snap"
    os.makedirs(save_dir, exist_ok=True)
    saved_count = 0
    with LOCK:
        for key, img in LIVE.items():
            safe_name = key.replace("/", "_").replace(".", "_")
            path = os.path.join(save_dir, f"{safe_name}.jpg")
            img.save(path)
            saved_count += 1
            print(f"[💾] 已保存快照: {path}")
            
    print("=" * 66)
    if saved_count > 0:
        print(f"[💡 提示] 请打开当前目录下的 {save_dir} 文件夹查看图片，")
        print(f"         以确认画面内容是否正确（如：哪台是左腕，哪台是右腕）。")
    
    print("[结论] 三路全部收到画面 ✅" if ok else "[结论] 有端口未收到画面 ❌")

if __name__ == "__main__":
    main()
