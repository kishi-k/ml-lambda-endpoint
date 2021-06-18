#!/bin/sh
if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
  exec /usr/bin/aws-lambda-rie /opt/conda/bin/python -m awslambdaric
else
  exec /opt/conda/bin/python -m aws-lambda-ric
fi     