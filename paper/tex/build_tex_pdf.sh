#!/bin/sh

echo "Run platex pbibtex platex platex dvipdfmx."

cd /mnt/d/0ngoing/thesis/repo/paper/tex
platex eps_forecast
pbibtex eps_forecast
platex eps_forecast
platex eps_forecast
dvipdfmx eps_forecast