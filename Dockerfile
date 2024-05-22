FROM public.ecr.aws/lambda/python:3.10

WORKDIR ${LAMBDA_TASK_ROOT}


# Install the function's dependencies using file requirements.txt
COPY requirements.txt  .


RUN yum install file-devel tesseract

RUN pip3 install --no-cache-dir --upgrade pip && \ 
    pip3 install --no-cache-dir -r requirements.txt


COPY . .


# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD ["lambda_function.lambda_handler"]