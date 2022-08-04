#!/bin/bash

## Log and model folders can grow very large. This script creates symlinks to outsource them.
## It requires a target directory as single parameter, e.g:
##
## bash setup_symlinks.sh $HOME/experiments/my-project


TARGET_DIR=$1
[[ -z "$TARGET_DIR" ]] && { echo "Error: No target directory was provided"; exit 1; }
DIRS=( "logs/experiments" "logs/evaluations" "logs/debugs" "models" )

echo "symlink to $TARGET_DIR..."
for d in "${DIRS[@]}"
do
    echo "symlink: $d"
    mkdir -p "$TARGET_DIR/$d"
    ln -s -T "$TARGET_DIR/$d" "$d"
done
