#!/usr/bin/env bash
echo "########## 1. 视频设备 & USB 拓扑 ##########"
v4l2-ctl --list-devices

echo ""
echo "########## 2. 每个节点的稳定标识 ##########"
for d in /sys/class/video4linux/video*; do
  node=$(basename "$d")
  echo "--- /dev/$node ---"
  echo "  name    : $(cat "$d/name" 2>/dev/null)"
  dev=$(readlink -f "$d/device" 2>/dev/null)
  echo "  sysfs   : $dev"
  cur=$dev
  for i in 1 2 3 4 5 6; do
    if [ -f "$cur/serial" ]; then
      echo "  serial  : $(cat "$cur/serial" 2>/dev/null)"
      echo "  vid_pid : $(cat "$cur/idVendor" 2>/dev/null):$(cat "$cur/idProduct" 2>/dev/null)"
      echo "  bcd     : $(cat "$cur/bcdDevice" 2>/dev/null)"
      break
    fi
    cur=$(dirname "$cur")
  done
done

echo ""
echo "########## 3. RealSense 设备 ##########"
rs-enumerate-devices 2>/dev/null | grep -E "Name|Serial|usb" || echo "(无 rs-enumerate-devices 或未接 RealSense)"

echo ""
echo "########## 4. 各捕获节点支持的分辨率/帧率 ##########"
for d in /sys/class/video4linux/video*; do
  node=$(basename "$d")
  name=$(cat "$d/name" 2>/dev/null)
  echo "$name" | grep -qi "metadata" && continue
  echo "--- /dev/$node ($name) ---"
  v4l2-ctl -d "/dev/$node" --list-formats-ext 2>/dev/null | head -30
done

echo ""
echo "########## 5. 机器人 IP ##########"
hostname -I 2>/dev/null || ip -brief addr

echo ""
echo "########## 6. 端口占用 ##########"
for p in 5555 5556 5557 6000 6001; do
  ss -tlnp 2>/dev/null | grep -q ":$p " && echo "端口 $p : [已占用]" || echo "端口 $p : [空闲]"
done

echo ""
echo "########## 7. USB 物理树 ##########"
lsusb -t 2>/dev/null || echo "(lsusb -t 不可用)"
