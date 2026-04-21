# Script to build (and push) docker images.

# Note that this assumes that an appropriate multiplatform docker builder is available.
# i.e. it may be necessary to run:
# > docker buildx create --use
# first.

usage() {
    echo "Usage:" 1>&2
    echo "bash $0 [-h] [-t tag<string>] [-p platform<string>,platform<string>]" 1>&2
    echo "  -h: print this help message and exit" 1>&2
    echo "  -t: specify a tag name (defaults to latest)" 1>&2
    echo "  -p: comma separated list of platforms (defaults to current platform)" 1>&2
    echo "  -f: force a rebuild (i.e. use --no-cache)" 1>&2
}

error() {
    usage
    exit 1
}

# realpath not available by default on macs so define it here
realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

full_path=$(realpath $0)
script_path=$(dirname $full_path)

# parse the arguments
TAG=''
PLATFORMS=''
BUILDER=''
FORCE=''

while getopts ":t:p:fh" opt; do
    case $opt in
        h )
           usage
           exit 0
           ;;
        t )
           TAG=${OPTARG}
           ;;
        p )
           PLATFORMS="--platform ${OPTARG}"
           ;;
        f )
            FORCE='--no-cache'
            ;;
        : )
           echo "ERROR: -${OPTARG} requires an argument." 1>&2
           error
           ;;
        * )
           echo "ERROR: unknown option -${OPTARG}." 1>&2
           error
           ;;
    esac
done

shift $((OPTIND - 1))

PTAG=''
if [ -z "$PLATFORMS" ]; then
    PROC=`uname -m`
    if [ "$PROC" == "x86_64" ]; then
        PTAG="amd64"
    elif [ "$PROC" == "arm64" ]; then
        PTAG="arm64"
    fi
    PLATFORMS='--provenance false'
else
    BUILDER='buildx'
fi

# if no tag is specified default to latest
if [ -z "$TAG" ]; then
    TAG="latest"
    if [ "$PTAG" ]; then
        TAG="${TAG}-${PTAG}"
    fi
fi

echo "Building:"
echo "  Dockerfile"
echo "  with tag ghcr.io/cianwilson/fenics-sz:$TAG"
if [ "$PLATFORMS" ]; then
  echo "  with $PLATFORMS"
fi

cd $script_path
docker $BUILDER build $FORCE --file Dockerfile \
                      --tag ghcr.io/cianwilson/fenics-sz:$TAG $PLATFORMS --push .

