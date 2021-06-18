# Serverless ML API Sample (by AWS Lambda + Docker )

## Sample Assets

- vehicle
Vehicle Detection model by tensorflow.  
Please download below as `/vehicle/app/tmp`.

https://github.com/kishi-k/vehicle_counting_tensorflow


## How to build 

```
docker build -t {tag_name} .
```

## Run local test using aws-lambda-rie


```
docker run  -p 9000:8080 --entrypoint /usr/bin/aws-lambda-rie  --name serverless  --rm serverless-function:latest  /usr/local/bin/python -m awslambdaric app.handler
```

## Sample

Please refer to [this Notebook](./test_lambda_endpoint.ipynb)