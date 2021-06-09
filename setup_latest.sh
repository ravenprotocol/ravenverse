rm -rf ravcom
rm -rf ravop
rm -rf ravml
rm -rf ravsock
rm -rf ravjs
rm -rf ravviz

pip uninstall ravcom -y
pip install git+https://github.com/ravenprotocol/ravcom.git --force

pip uninstall ravop -y
pip install git+https://github.com/ravenprotocol/ravop.git --force

pip uninstall ravml -y
pip install git+https://github.com/ravenprotocol/ravml.git --force

pip uninstall ravsock -y
pip install git+https://github.com/ravenprotocol/ravsock.git --force

pip uninstall ravviz -y
pip install git+https://github.com/ravenprotocol/ravviz.git --force

git clone https://github.com/ravenprotocol/ravjs.git
