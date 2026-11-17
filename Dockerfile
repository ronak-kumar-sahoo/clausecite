FROM public.ecr.aws/lambda/python:3.12
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV HF_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
COPY src/ ${LAMBDA_TASK_ROOT}/
COPY web/ ${LAMBDA_TASK_ROOT}/web/
COPY chroma_db/ ${LAMBDA_TASK_ROOT}/chroma_db/
CMD ["lambda_handler.handler"]
