#!/bin/sh
# Fetch evaluation datasets into datasets/ (git-ignored).
#
# License note: these are research datasets over commercial recordings.
# Annotations are used with citation (Krebs et al. 2013 for Ballroom); audio
# must never be redistributed or committed. GTZAN's canonical audio mirror
# (marsyas.info) is dead; we use Ballroom as the primary corpus.
#
# RWC 2.0 (sections eval): audio is CC BY-NC 4.0 (Zenodo record 17177919) —
# non-commercial research use with citation of Goto et al. (ISMIR 2002/2003)
# and Balke et al. (TISMIR 2026, doi:10.5334/tismir.326). Structure
# annotations (AIST CHORUS) are research-use-only per their README.
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

# RWC 2.0: 13.4 GB monolithic zip; only the popular-music subset (100 WAVs,
# ~4 GB) is extracted. Structure annotations come from the archive repo.
if [ ! -d datasets/rwc-audio ]; then
    echo "==> RWC 2.0 audio (13.4 GB zip, CC BY-NC 4.0 — see header)"
    curl -L -C - --retry 3 -o datasets/RWC-CGJPR.zip \
        "https://zenodo.org/records/17177919/files/RWC-CGJPR.zip?download=1"
    actual=$(md5sum datasets/RWC-CGJPR.zip 2>/dev/null || md5 -q datasets/RWC-CGJPR.zip)
    [ "${actual%% *}" = "272309ed06121e6ed2ac5c8f55664baa" ] || {
        echo "RWC zip checksum mismatch (got: $actual)"; exit 1;
    }
    mkdir -p datasets/rwc-audio
    unzip -q -j -o datasets/RWC-CGJPR.zip 'RWC_OpenAccess_WAV/RWC-P/*.wav' \
        -d datasets/rwc-audio/
else
    echo "==> RWC popular-music audio already present"
fi

if [ ! -d datasets/rwc-annotations-archive ]; then
    echo "==> RWC structure annotations (rwc-music/rwc-annotations-archive)"
    git clone --depth 1 --quiet \
        https://github.com/rwc-music/rwc-annotations-archive datasets/rwc-annotations-archive
else
    echo "==> RWC annotations archive already present"
fi

echo "Done. Evaluate with:"
echo "  cargo run --release --example eval -- datasets/BallroomData datasets/ballroom-annotations"
echo "  cargo run --release --example eval_sections -- datasets/rwc-audio datasets/rwc-annotations-archive/AIST_RWC-MDB-P-2001_CHORUS"
