#FROM tensorflow/tensorflow:1.15.4-gpu-py3-jupyter 
FROM tensorflow/tensorflow:1.15.4-py3-jupyter 
RUN apt-get update
RUN apt install -y git
RUN apt install -y vim
RUN apt install -y libgl1-mesa-glx
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

