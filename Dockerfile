FROM tensorflow/tensorflow:1.15.4-gpu

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda --version

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
RUN mkdir /home/mrcnn
WORKDIR /home/mrcnn
COPY . .
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH
SHELL ["conda", "run", "-n", "cell_mrcnn", "/bin/bash", "-c"]
#RUN apt-get update && \
#    apt-get upgrade -y && \
#    apt-get install -y git

RUN pip install scikit-image
RUN pip install streamlit-ace
#RUN python /home/mrcnn/setup.py install
#CMD streamlit run app/cell_app.py --server.port 6333
CMD streamlit run app/uploader.py
