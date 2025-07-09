#!/bin/bash
# This script is used to stage, commit, and push changes to the Git repository.
if git add . && git commit -m "$1" && git push origin main; then
  echo "Changes pushed successfully"
else
  echo "Failed to push changes"
  exit 1
fi
