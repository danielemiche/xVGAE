#!/usr/bin/env bash
# Build ORCA (Hocevar & Demsar 2014), the 4-node orbit counter behind the orbit MMD, into xvgae/orca/orca.
#
# ORCA is GPL-3.0 (github.com/thocevar/orca), so it is NOT part of this MIT repository: this script downloads the copy
# that GraphRNN (You et al. 2018) ships, at a pinned commit, checks its sha256, adds the two missing `return 0;` of
# writeResults / writeEdgeResults (the counts do not change), and compiles it.
#     bash scripts/build_orca.sh                  # download + build
#     bash scripts/build_orca.sh /path/to/orca.cpp /path/to/orca.h   # build from local files (no download)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/xvgae/orca"
COMMIT=1ef475d957414d7c0bf8c778a1d44cb52dd7829b
URL="https://raw.githubusercontent.com/snap-stanford/GraphRNN/$COMMIT/eval/orca"
SHA_CPP=c95687412fe8fcc0e19d250f9274f3b0a803b5bc01ecf0f2d2276f93f518b450
SHA_H=4a4851a1f42f7b929e49ae413909bd460191ce8e1951fb8e1cb3e589cccc5f78
mkdir -p "$DEST"
if [ $# -ge 2 ]; then
  cp "$1" "$DEST/orca.cpp"; cp "$2" "$DEST/orca.h"
else
  curl -sSfL "$URL/orca.cpp" -o "$DEST/orca.cpp"
  curl -sSfL "$URL/orca.h" -o "$DEST/orca.h"
  echo "$SHA_CPP  $DEST/orca.cpp" | sha256sum -c -
  echo "$SHA_H  $DEST/orca.h" | sha256sum -c -
fi
python3 - "$DEST/orca.cpp" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
for fn in ("int writeResults(", "int writeEdgeResults("):
    i = s.index(fn); j = s.index("fout.close();", i); k = s.index("}", j)
    if "return 0;" not in s[j:k]:
        s = s[:j] + "fout.close();\n    return 0;\n" + s[k:]
open(p, "w").write(s)
PY
g++ -O2 -std=c++11 "$DEST/orca.cpp" -o "$DEST/orca"
echo "built $DEST/orca"
