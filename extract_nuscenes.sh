#!bin/bash
cd /data1/work/irdali.durrani/nuscenes_data

for f in *.tgz; do
    tar --skip-old-files \
        --exclude='LIDAR_TOP' \
        --exclude='RADAR_*' \
        -xvzf "$f"
done

unzip can_bus.zip -d ./