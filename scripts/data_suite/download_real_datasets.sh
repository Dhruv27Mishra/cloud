#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REAL_DIR="$ROOT/datasets/real"

mkdir -p "$REAL_DIR/google_cluster_2011/raw" "$REAL_DIR/azure_vm_2019/raw" "$REAL_DIR/alibaba_cluster_2018/raw_archives"

echo "[1/3] Google cluster bucket listing (requires gcloud auth)"
if command -v gcloud >/dev/null 2>&1; then
  gcloud storage ls gs://clusterdata-2011-2 > "$REAL_DIR/google_cluster_2011/bucket_listing.txt" || true
  echo "Saved: $REAL_DIR/google_cluster_2011/bucket_listing.txt"
else
  echo "gcloud not found; install Google Cloud SDK to download Google trace."
fi

echo "[2/3] Azure links manifest"
AZ_LINKS="$REAL_DIR/azure_vm_2019/AzurePublicDatasetLinksV2.txt"
curl -L "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzurePublicDatasetLinksV2.txt" -o "$AZ_LINKS"
echo "Saved: $AZ_LINKS"

echo "[3/3] Azure sample shards"
head -n 12 "$AZ_LINKS" | while IFS= read -r u; do
  [[ -z "${u}" ]] && continue
  f="$(basename "$u")"
  curl -L "$u" -o "$REAL_DIR/azure_vm_2019/raw/$f" || true
done
echo "Saved sample files under: $REAL_DIR/azure_vm_2019/raw"

cat <<EOF
Alibaba trace (v2018) is survey-gated in official docs:
  http://alibabadeveloper.mikecrm.com/BdJtacN
After download, place archives under:
  $REAL_DIR/alibaba_cluster_2018/raw_archives
EOF
