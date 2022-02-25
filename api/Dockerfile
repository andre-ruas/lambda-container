FROM public.ecr.aws/lambda/python:3.8

COPY inference.py ${LAMBDA_TASK_ROOT}
COPY train.py ${LAMBDA_TASK_ROOT}
COPY saveData.py ${LAMBDA_TASK_ROOT}

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

