#!/bin/sh
echo "Run platex pbibtex platex platex dvipdfmx."

cd /mnt/d/0ngoing/thesis/repo/paper/tex/abstract
platex abstract
pbibtex abstract
platex abstract
platex abstract
dvipdfmx abstract