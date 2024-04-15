FROM python:3.10
USER root

RUN apt-get update && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get -y install software-properties-common && \
    pip install --upgrade pip && \
    apt-get install -y sudo wget vim curl gawk make gcc

RUN mkdir ~/.config  && \
    mkdir ~/.config/git && \
    touch ~/.config/git/ignore && \
    echo ".DS_Store" > ~/.config/git/ignore && \
    echo ".env" >> ~/.config/git/ignore && \
    echo ".pre-commit-config.yaml" >> ~/.config/git/ignore && \
    echo ".pyproject.toml" >> ~/.config/git/ignore && \
    echo ".vscode" >> ~/.config/git/ignore && \
    echo "test.ipynb" >> ~/.config/git/ignore && \
    echo "image" >> ~/.config/git/ignore && \
    echo "article.md" >> ~/.config/git/ignore

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501