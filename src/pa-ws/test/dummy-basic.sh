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
INFO | BASIC_STATUS {"state": "progress", "stageNumber": $1, "stageName": "$2", "counter": $3, "timeToNextStatus": $4, "stageWeighting": "$STAGE_WEIGHTING", "timestamp": "$TIMESTAMP"}
EOF
}

report_status 0 "init" 0 1

function count {
    for i in $(seq 1 ${STATUS_COUNT}); do
        sleep $STATUS_DELAY_SECONDS
        report_status 1 "basic" $i $STATUS_DELAY_SECONDS
    done
}

report_status 2 "fini" 0 1

count

# Close the file-descriptor, to tell the parent we are done.
# This might be required, so we need to think about how to avoid hanging.
# eval is needed:
#   https://stackoverflow.com/questions/8295908/how-to-use-a-variable-to-indicate-a-file-descriptor-in-bash
#echo "Closing FD '$FD'" >&$FD
#eval "exec $FD>&-"
# I am hoping that we do not need this, as it's hard to guarantee. ~cdunn
