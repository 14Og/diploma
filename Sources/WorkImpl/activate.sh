#!/bin/bash

set -e
VENV_NAME=".venv"
PYTHON_VERSION=$(python3 -V | cut -d " " -f 2 | cut -d "." -f 1,2)
REQUIRED_VERSION="3.10"

 if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
        echo "Latest python3 version found: ${PYTHON_VERSION}. OK"

 else
        echo "Latest python3 version is less than required: ${PYTHON_VERSION} vs ${REQUIRED_VERSION}. EXIT"
        exit 1
 fi

if [[ $(ps -p $$ | grep zsh) ]]; then
    echo "Running in zsh"
    VENV_ACTIVATE=". ${VENV_NAME}/bin/activate"
else
    echo "Running in bash"
    VENV_ACTIVATE="source ${VENV_NAME}/bin/activate"
fi

pkgs='dvipng texlive-latex-extra texlive-fonts-recommended cm-super fonts-noto-cjk pandoc'
# install=false
# for pkg in $pkgs; do
#     status="$(dpkg-query -W --showformat='${db:Status-Status}' "$pkg" 2>&1)"
#     echo $status
#     if [ ! $? = 0 ] || [ ! "$status" = installed ]; then
#         install=true
#         break
#     # else
#         # echo "$pkg is installed"
#     fi    
# done
# if "$install"; then
sudo apt install $pkgs
# fi

python$PYTHON_VERSION -m venv $VENV_NAME
eval "$VENV_ACTIVATE"
pip install -r requirements.txt
