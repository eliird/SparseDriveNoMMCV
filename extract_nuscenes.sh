#!/bin/bash
# Extract all nuScenes data files and map expansions

cd /data1/work/irdali.durrani/nuscenes_data

echo "========================================="
echo "NuScenes Data Extraction Script"
echo "========================================="
echo ""

# 1. Extract main nuScenes data files (.tgz)
echo "Step 1: Extracting nuScenes trainval data..."
for f in v1.0-trainval*.tgz; do
    if [ -f "$f" ]; then
        echo "  - Extracting $f..."
        tar --skip-old-files \
            --exclude='RADAR_*' \
            -xzf "$f"
    fi
done

# 2. Extract metadata
echo ""
echo "Step 2: Extracting metadata..."
if [ -f "v1.0-trainval_meta.tgz" ]; then
    echo "  - Extracting v1.0-trainval_meta.tgz..."
    tar --skip-old-files -xzf v1.0-trainval_meta.tgz
fi

# 3. Extract CAN bus data
echo ""
echo "Step 3: Extracting CAN bus data..."
if [ -f "can_bus.zip" ]; then
    echo "  - Extracting can_bus.zip..."
    unzip -o can_bus.zip -d ./
fi

# 4. Extract map expansion files
echo ""
echo "Step 4: Extracting map expansion files..."
for f in nuScenes-map-expansion-v*.zip; do
    if [ -f "$f" ]; then
        echo "  - Extracting $f..."
        unzip -o "$f" -d ./
    fi
done

# 5. Organize map expansion files into correct directory structure
echo ""
echo "Step 5: Organizing map expansion files..."
mkdir -p maps/expansion

# Move expansion files from v1.3 format (expansion/*.json -> maps/expansion/)
if [ -d "expansion" ]; then
    echo "  - Moving expansion/*.json to maps/expansion/"
    mv expansion/*.json maps/expansion/ 2>/dev/null
    rmdir expansion 2>/dev/null
fi

# Move basemap files
if [ -d "basemap" ]; then
    echo "  - Moving basemap to maps/"
    mv basemap maps/ 2>/dev/null
fi

# Move prediction files
if [ -d "prediction" ]; then
    echo "  - Moving prediction to maps/"
    mv prediction maps/ 2>/dev/null
fi

# Handle v1.1 format (maps/*.json already in correct location, but need to ensure expansion subdir)
if [ -d "maps" ] && [ ! -d "maps/expansion" ]; then
    mkdir -p maps/expansion
    mv maps/*.json maps/expansion/ 2>/dev/null
fi

# Handle v1.2 format (nuScenes-map-expansion-v1.2/*.json)
if [ -d "nuScenes-map-expansion-v1.2" ]; then
    echo "  - Moving nuScenes-map-expansion-v1.2/*.json to maps/expansion/"
    mv nuScenes-map-expansion-v1.2/*.json maps/expansion/ 2>/dev/null
    rm -rf nuScenes-map-expansion-v1.2 2>/dev/null
fi

# 6. Verification
echo ""
echo "========================================="
echo "Verification"
echo "========================================="

# Check samples/LIDAR_TOP
if [ -d "samples/LIDAR_TOP" ]; then
    lidar_count=$(find samples/LIDAR_TOP -name "*.pcd.bin" | wc -l)
    echo "✓ LIDAR_TOP samples found ($lidar_count files)"
else
    echo "✗ LIDAR_TOP samples not found"
fi

# Check cameras
if [ -d "samples/CAM_FRONT" ]; then
    cam_count=$(find samples/CAM_FRONT -name "*.jpg" | wc -l)
    echo "✓ Camera samples found ($cam_count files in CAM_FRONT)"
else
    echo "✗ Camera samples not found"
fi

# Check sweeps/LIDAR_TOP
if [ -d "sweeps/LIDAR_TOP" ]; then
    sweep_count=$(find sweeps/LIDAR_TOP -name "*.pcd.bin" | wc -l)
    echo "✓ LIDAR sweeps found ($sweep_count files)"
else
    echo "✗ LIDAR sweeps not found"
fi

# Check metadata
if [ -d "v1.0-trainval" ]; then
    meta_count=$(ls v1.0-trainval/*.json 2>/dev/null | wc -l)
    echo "✓ v1.0-trainval metadata found ($meta_count JSON files)"
else
    echo "✗ v1.0-trainval metadata not found"
fi

# Check CAN bus
if [ -d "can_bus" ]; then
    echo "✓ CAN bus data found"
else
    echo "✗ CAN bus data not found"
fi

# Check map expansion
if [ -d "maps/expansion" ]; then
    map_count=$(ls maps/expansion/*.json 2>/dev/null | wc -l)
    if [ $map_count -gt 0 ]; then
        echo "✓ Map expansion files found ($map_count JSON files)"
        ls maps/expansion/*.json 2>/dev/null | xargs -n1 basename | sed 's/^/    - /'
    else
        echo "✗ Map expansion files not found in maps/expansion/"
    fi
else
    echo "✗ maps/expansion/ directory not found"
fi

# Check basemap
if [ -d "maps/basemap" ]; then
    basemap_count=$(ls maps/basemap/*.png 2>/dev/null | wc -l)
    echo "✓ Basemap files found ($basemap_count PNG files)"
else
    echo "⚠ Basemap files not found (optional)"
fi

echo ""
echo "========================================="
echo "Extraction complete!"
echo "========================================="