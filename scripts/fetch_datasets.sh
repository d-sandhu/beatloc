#!/bin/sh
# Fetch evaluation datasets into datasets/ (git-ignored).
#
# License note: these are research datasets over commercial recordings.
# Annotations are used with citation (Krebs et al. 2013 for Ballroom); audio
# must never be redistributed or committed. GTZAN's canonical audio mirror
# (marsyas.info) is dead; we use Ballroom as the primary corpus.
set -eu

mkdir -p datasets

if [ ! -d datasets/BallroomData ]; then
    echo "==> Ballroom audio (~1.4 GB, ISMIR2004 tempo contest)"
    curl -L -C - --retry 3 -o datasets/ballroom-audio.tar.gz \
        https://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz
    (cd datasets && tar xzf ballroom-audio.tar.gz)
else
    echo "==> Ballroom audio already present"
fi

if [ ! -d datasets/ballroom-annotations ]; then
    echo "==> Ballroom beat/downbeat annotations (CPJKU/BallroomAnnotations)"
    git clone --depth 1 --quiet \
        https://github.com/CPJKU/BallroomAnnotations datasets/ballroom-annotations
else
    echo "==> Ballroom annotations already present"
fi

echo "Done. Evaluate with:"
echo "  cargo run --release --example eval -- datasets/BallroomData datasets/ballroom-annotations"
