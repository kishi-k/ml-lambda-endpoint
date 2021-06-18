# Define function directory
ARG FUNCTION_DIR="/function"

FROM tensorflow/tensorflow:1.14.0-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install aws-lambda-cpp build dependencies
RUN apt-get update && \
  apt-get install -y \
  g++ \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev \
  libsm6 \
  libxrender1 \
  libxtst6 \
  unzip \
  # Install for Doclerfile
  python3-dev \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgl1-mesa-dev \
  python3-tk


# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Create function directory
RUN mkdir -p ${FUNCTION_DIR}

# Copy function code
COPY app/app.py ${FUNCTION_DIR}/app.py 
COPY app/tmp/ ${FUNCTION_DIR}/tmp/ 

RUN mv ${FUNCTION_DIR}/tmp/*  ${FUNCTION_DIR} 

COPY requirements.txt  /

# Install the runtime interface client
RUN pip install --upgrade pip && \  
    pip install awslambdaric && \
    pip install --target ${FUNCTION_DIR} --no-cache-dir -r requirements.txt 


# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}


# for TEST
COPY aws-lambda-rie /usr/bin/
RUN chmod 755 /usr/bin/aws-lambda-rie

# ENTRYPOINT [ “/entry_script.sh” ]

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "app.handler" ]
