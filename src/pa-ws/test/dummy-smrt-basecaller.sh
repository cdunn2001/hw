#!/bin/bash
# https://stackoverflow.com/a/14203146

POSITIONAL_ARGS=()
FD=2

while [[ $# -gt 0 ]]; do
  case $1 in
    --status-fd)
      FD="$2"
      shift # past argument
      shift # past value
      ;;
    --logfilter)
      LOG_FILTER="$2"
      shift # past argument
      shift # past value
      ;;
    --logoutput)
      LOG_OUTPUT="$2"
      shift # past argument
      shift # past value
      ;;
    --outputbazfile)
      OUTPUT_BAZ_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    --outputtrcfile)
      OUTPUT_TRC_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    --config)
      CONFIG="$2"
      shift # past argument
      shift # past value
      ;;
    --maxFrames)
      MAX_FRAMES="$2"
      shift # past argument
      shift # past value
      ;;
    --numWorkerThreads)
      NUM_WORKER_THREADS="$2"
      shift # past argument
      shift # past value
      ;;
    --version) # maybe?
      VERSION=1
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      #exit 1 # Fine.
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

# Optional env-vars:
: "${STATUS_COUNT:=0}"
: "${STATUS_DELAY_SECONDS:=0.0}"

TIMESTAMP="20220223T146198.099Z" # arbitrary
STAGE_WEIGHTING="[0, 100, 0]"

function report_status {
    # ARGS: number, name, counter, next
    # Not reported: counterMax
    # Do we need "timestamp"?
    cat >&$FD << EOF
INFO | SMRT_BASECALLER_STATUS {"state": "progress", "stageNumber": $1, "stageName": "$2", "counter": $3, "timeToNextStatus": $4, "stageWeighting": "$STAGE_WEIGHTING", "timestamp": "$TIMESTAMP"}
EOF
}

report_status 0 "init" 0 1

function count {
    for i in $(seq 1 ${STATUS_COUNT}); do
        sleep $STATUS_DELAY_SECONDS
        report_status 1 "baz2bam" $i $STATUS_DELAY_SECONDS
    done
}

report_status 2 "fini" 0 1

count

touch ${LOG_OUTPUT}
touch ${OUTPUT_BAZ_FILE}
touch ${OUTPUT_TRC_FILE}
