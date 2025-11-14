cenes_download.sh

# List of files to download
URLS=(
"https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval10_blobs.tgz"
"https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval09_blobs.tgz"
"https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval08_blobs.tgz"
"https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_blobs.tgz"
"https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval06_blobs.tgz"
"https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_blobs.tgz"
"https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval04_blobs.tgz"
"https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval03_blobs.tgz"
"https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval02_blobs.tgz"
"https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz"
"https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz"
"https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/can_bus.zip"
)

# Create target directory
mkdir -p /data1/work/irdali.durrani/nuscenes_data
cd /data1/work/irdali.durrani/nuscenes_data

# Download each file with progress bar
for url in "${URLS[@]}"; do
	    echo "Downloading $url..."
	        wget -c --progress=bar:force:noscroll "$url" -O "$(basename "$url")"
		    echo
	    done
	    cd ../../
