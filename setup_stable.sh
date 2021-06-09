pip uninstall ravcom -y
pip install git+https://github.com/ravenprotocol/ravcom.git@0.1-alpha

pip uninstall ravop -y
pip install git+https://github.com/ravenprotocol/ravop.git@0.1-alpha

pip uninstall ravml -y
pip install git+https://github.com/ravenprotocol/ravml.git@0.1-alpha

pip uninstall ravsock -y
pip install git+https://github.com/ravenprotocol/ravsock.git@0.1-alpha

pip uninstall ravviz -y
pip install git+https://github.com/ravenprotocol/ravviz.git0.1-alpha

wget https://github.com/ravenprotocol/ravjs/archive/refs/tags/0.1-alpha.zip
unzip 0.1-alpha.zip
rm 0.1-alpha.zip
mv ravjs-0.1-alpha ravjs
