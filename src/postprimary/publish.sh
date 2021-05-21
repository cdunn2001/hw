#!/usr/bin/env bash
set -ex
echo We are building branch ${bamboo_planRepository_branchName}

if [ -z "${bamboo_globalBuildNumber}" ]; then
  echo This script is for use by Bamboo.
  exit 1
fi

case "$bamboo_planRepository_branchName" in
  master)
    moduleVersion="${bamboo_release_version}-${bamboo_globalBuildNumber}"
    ;;
  develop)
    moduleVersion="develop-${bamboo_globalBuildNumber}"
    ;;
  release/prep)
    moduleVersion="rp-${bamboo_globalBuildNumber}"
    ;;
  release/${bamboo_release_version})
    moduleVersion="${bamboo_release_version}-rc-${bamboo_globalBuildNumber}"
    ;;
  *)
    echo "Branch is ${bamboo_planRepository_branchName}."
    echo "Only publish develop and master builds."
    exit 0
    ;;
esac

if [[ -e bin/baz2bam ]]; then
  artifactbase="bin"
  baz2bam="${artifactbase}/baz2bam"
  bazviewer="${artifactbase}/bazviewer"
  bam2bam="${artifactbase}/bam2bam"
  recalladapters="${artifactbase}/recalladapters"
elif [[ -e bin/Release/baz2bam ]]; then
  artifactbase="bin/Release"
  baz2bam="${artifactbase}/baz2bam"
  bazviewer="${artifactbase}/bazviewer"
  bam2bam="${artifactbase}/bam2bam"
  recalladapters="${artifactbase}/recalladapters"
elif [[ -e bin/Release/application/baz2bam ]]; then
  artifactbase="bin/Release/application"
  baz2bam="${artifactbase}/baz2bam"
  bazviewer="bin/Release/pacbio-primary/bazviewer/bazviewer"
  bam2bam="${artifactbase}/bam2bam"
  recalladapters="${artifactbase}/recalladapters"
elif [[ -e baz2bam ]]; then
  artifactbase="."
  baz2bam="${artifactbase}/baz2bam"
  bazviewer="${artifactbase}/bazviewer"
  bam2bam="${artifactbase}/bam2bam"
  recalladapters="${artifactbase}/recalladapters"
else
  echo "[ERROR] cannot determine where the artifacts are"
  exit 1
fi
rsync -aP ${baz2bam} ${bazviewer} ${bam2bam} ${recalladapters} \
    /mnt/software/p/ppa/${moduleVersion}/
    cat > /mnt/software/modulefiles/ppa/${moduleVersion} << EOF
#%Module1.0#####################################################################
##
conflict ppa
module-whatis "Access PPA ${moduleVersion}"
prepend-path  PATH /mnt/software/p/ppa/${moduleVersion}
EOF
case "$bamboo_planRepository_branchName" in
  master)
    ln -sfn ${moduleVersion} /mnt/software/modulefiles/ppa/master
    ln -sfn ${moduleVersion} /mnt/software/modulefiles/ppa/${bamboo_release_version}
    ;;
  develop)
    ln -sfn ${moduleVersion} /mnt/software/modulefiles/ppa/develop
    ln -sfn ${moduleVersion} /mnt/software/modulefiles/ppa/mainline
    ;;
  release/prep)
    ln -sfn ${moduleVersion} /mnt/software/modulefiles/ppa/rp
    ln -sfn ${moduleVersion} /mnt/software/modulefiles/ppa/${bamboo_release_version}
    ;;
  release/${bamboo_release_version})
    ln -sfn ${moduleVersion} /mnt/software/modulefiles/ppa/${bamboo_release_version}-rc
    ;;
  *)
    exit 0
    ;;
esac
